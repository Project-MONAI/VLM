# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import shutil
import tempfile
import time
import traceback
from collections import OrderedDict
from urllib.parse import quote_plus

import ctk
import qt
import SampleData
import SimpleITK as sitk
import sitkUtils
import slicer
import vtk
import vtkSegmentationCore
from RadCoPilotLib import RadCoPilotClient
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


class RadCoPilot(ScriptedLoadableModule):
    '''RadCoPilot class.'''
    def __init__(self, parent):
        '''Initialize the RadCoPilot class.'''
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("RadCoPilot")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Radiology CoPilot")]
        self.parent.dependencies = []
        self.parent.contributors = ["3D Slicer", "NVIDIA"]
        self.parent.helpText = _("Radiology CoPilot 3D Slicer Module.")
        self.parent.acknowledgementText = _("Developed by 3D Slicer and NVIDIA developers")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.initializeAfterStartup)

    def initializeAfterStartup(self):
        '''Initialize additional components after application startup is complete.'''
        if not slicer.app.commandOptions().noMainWindow:
            self.settingsPanel = RadCoPilotSettingsPanel()
            slicer.app.settingsDialog().addPanel("RadCoPilot", self.settingsPanel)


class _ui_RadCoPilotSettingsPanel:
    def __init__(self, parent):
        '''Initialize the RadCoPilot settings panel.'''
        vBoxLayout = qt.QVBoxLayout(parent)

        # settings
        groupBox = ctk.ctkCollapsibleGroupBox()
        groupBox.title = _("RadCoPilot")
        groupLayout = qt.QFormLayout(groupBox)

        serverUrl = qt.QLineEdit()
        groupLayout.addRow(_("Server address:"), serverUrl)
        parent.registerProperty("RadCoPilot/serverUrl", serverUrl, "text", str(qt.SIGNAL("textChanged(QString)")))

        serverUrlHistory = qt.QLineEdit()
        groupLayout.addRow(_("Server address history:"), serverUrlHistory)
        parent.registerProperty(
            "RadCoPilot/serverUrlHistory", serverUrlHistory, "text", str(qt.SIGNAL("textChanged(QString)"))
        )

        fileExtension = qt.QLineEdit()
        fileExtension.setText(".nii.gz")
        fileExtension.toolTip = _("Default extension for uploading volumes")
        groupLayout.addRow(_("File Extension:"), fileExtension)
        parent.registerProperty(
            "RadCoPilot/fileExtension", fileExtension, "text", str(qt.SIGNAL("textChanged(QString)"))
        )

        vBoxLayout.addWidget(groupBox)
        vBoxLayout.addStretch(1)


class RadCoPilotSettingsPanel(ctk.ctkSettingsPanel):
    '''RadCoPilot Setting Panel class.'''
    def __init__(self, *args, **kwargs):
        '''Initialize RadCoPilot Setting Panel class.'''
        ctk.ctkSettingsPanel.__init__(self, *args, **kwargs)
        self.ui = _ui_RadCoPilotSettingsPanel(self)


class RadCoPilotWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    '''RadCoPilot Widget class.'''
    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.logic = None
        self._parameterNode = None
        self._volumeNode = None
        self._volumeNodes = []
        self._updatingGUIFromParameterNode = False

        self.info = {}
        self.current_sample = None
        self.samples = {}
        self.state = {
            "SegmentationModel": "",
            "DeepgrowModel": "",
            "ScribblesMethod": "",
            "CurrentStrategy": "",
            "CurrentTrainer": "",
        }
        self.file_ext = ".nii.gz"

        self.progressBar = None
        self.tmpdir = None
        self.timer = None

        self.optionsSectionIndex = 0
        self.optionsNameIndex = 0

    def setup(self):
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/RadCoPilot.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.tmpdir = slicer.util.tempDirectory("slicer-radcopilot")
        self.logic = RadCoPilotLogic()

        # Set icons and tune widget properties
        self.ui.serverComboBox.lineEdit().setPlaceholderText("enter server address or leave empty to use default")
        self.ui.fetchServerInfoButton.setIcon(self.icon("refresh-icon.png"))
        self.ui.uploadImageButton.setIcon(self.icon("upload.svg"))

        # start with button disabled
        self.ui.sendPrompt.setEnabled(False)
        self.ui.uploadImageButton.setEnabled(False)
        self.ui.outputText.setReadOnly(True)

        # Connections
        self.ui.fetchServerInfoButton.connect("clicked(bool)", self.onClickFetchInfo)
        self.ui.serverComboBox.connect("currentIndexChanged(int)", self.onClickFetchInfo)
        self.ui.sendPrompt.connect("clicked(bool)", self.onClickSendPrompt)
        self.ui.cleanOutputButton.connect("clicked(bool)", self.onClickCleanOutputButton)
        self.ui.uploadImageButton.connect("clicked(bool)", self.onUploadImage)

    def icon(self, name="RadCoPilot.png"):
        '''Get the icon for the RadCoPilot module.'''
        # It should not be necessary to modify this method
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", name)
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def updateServerSettings(self):
        '''Update the server settings based on the current UI state.'''
        self.logic.setServer(self.serverUrl())
        self.saveServerUrl()

    def serverUrl(self):
        '''Get the current server URL from the UI.'''
        serverUrl = self.ui.serverComboBox.currentText.strip()
        if not serverUrl:
            serverUrl = "http://localhost:8000"
        # return serverUrl.rstrip("/")
        return serverUrl

    def saveServerUrl(self):
        '''Save the current server URL to settings and update history.'''
        # self.updateParameterNodeFromGUI()

        # Save selected server URL
        settings = qt.QSettings()
        serverUrl = self.ui.serverComboBox.currentText
        settings.setValue("RadCoPilot/serverUrl", serverUrl)

        # Save current server URL to the top of history
        serverUrlHistory = settings.value("RadCoPilot/serverUrlHistory")
        if serverUrlHistory:
            serverUrlHistory = serverUrlHistory.split(";")
        else:
            serverUrlHistory = []
        try:
            serverUrlHistory.remove(serverUrl)
        except ValueError:
            pass

        serverUrlHistory.insert(0, serverUrl)
        serverUrlHistory = serverUrlHistory[:10]  # keep up to first 10 elements
        settings.setValue("RadCoPilot/serverUrlHistory", ";".join(serverUrlHistory))

        # self.updateServerUrlGUIFromSettings()

    def show_popup(self, title, message):
        '''Display a popup message box with the given title and message.'''
        msg_box = qt.QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def onClickFetchInfo(self):
        '''Handle the click event for fetching server information.'''
        start = time.time()

        try:
            self.updateServerSettings()
            info = self.logic.info()
            self.info = info

            print(f"Connected to RadCoPilot Server - Obtained info from server: {self.info}")
            self.show_popup("Information", "Connected to RadCoPilot Server")
            self.ui.sendPrompt.setEnabled(True)
            self.ui.uploadImageButton.setEnabled(True)
            # Updating model name
            self.ui.appComboBox.clear()
            self.ui.appComboBox.addItem(self.info)

        except AttributeError as e:
            slicer.util.errorDisplay(
                _("Failed to obtain server info. Please check your connection and try again."),
                detailedText=str(e)
            )
            return

        logging.info(f"Time consumed by fetch info: {time.time() - start:3.1f}")


    def onClickCleanOutputButton(self):
        '''Handle the click event for cleaning the output text.'''
        self.ui.outputText.clear()

    def onUploadImage(self, init_sample=True, session=False):
        '''Gets the volume and sen it to the server.'''
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        image_id = volumeNode.GetName()

        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            in_file = tempfile.NamedTemporaryFile(suffix=self.file_ext, dir=self.tmpdir).name
            self.current_sample = in_file
            self.reportProgress(5)

            start = time.time()
            slicer.util.saveNode(volumeNode, in_file)
            logging.info(f"Saved Input Node into {in_file} in {time.time() - start:3.1f}s")
            print(f'Current volume: {in_file}')
            self.reportProgress(30)
            self.reportProgress(100)

            self._volumeNode = volumeNode
            if init_sample:
                self.initSample({"id": image_id}, autosegment=False)
            qt.QApplication.restoreOverrideCursor()

            return True
        except BaseException as e:
            msg = f"Message: {e.msg}" if hasattr(e, "msg") else ""
            self.reportProgress(100)
            qt.QApplication.restoreOverrideCursor()
            return False

    def reportProgress(self, progressPercentage):
        '''Reports progress of an event.'''
        if not self.progressBar:
            self.progressBar = slicer.util.createProgressDialog(windowTitle=_("Wait..."), maximum=100)
        self.progressBar.show()
        self.progressBar.activateWindow()
        self.progressBar.setValue(progressPercentage)
        slicer.app.processEvents()

    def has_text(self, ui_text):
        '''Check if the given UI text element has any content.'''
        return len(ui_text.toPlainText()) < 1

    def onClickSendPrompt(self):
        '''Handle the click event for sending a prompt to the server.'''
        if not self.logic:
            return

        self.ui.outputText.clear()

        print(f"This is the image to send for analysis: {self.current_sample}")
        
        if self.has_text(self.ui.inputText):
            self.show_popup("Information", "Empty prompt")
            self.ui.outputText.clear()
        else:
            start = time.time()
            self.updateServerSettings()
            inText = self.ui.inputText.toPlainText()
            info = self.logic.getAnswer(inputText=inText, volumePath=self.current_sample)
            if info is not None:
                self.info = info
                self.ui.outputText.setText(info['choices'][0]['message']['content'])
            logging.info(f"Time consumed by fetch info: {time.time() - start:3.1f}")



class RadCoPilotLogic(ScriptedLoadableModuleLogic):
    '''RadCoPilot logic.'''
    def __init__(self, server_url=None, tmpdir=None, progress_callback=None):
        '''Initialize the RadCoPilot logic.'''
        ScriptedLoadableModuleLogic.__init__(self)

        self.server_url = server_url
        self.tmpdir = slicer.util.tempDirectory("slicer-radcopilot") if tmpdir is None else tmpdir
        self.progress_callback = progress_callback

    def __del__(self):
        '''del method declaration.'''
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def setServer(self, server_url=None):
        '''Set the server URL for the RadCoPilot client.'''
        self.server_url = server_url if server_url else "http://localhost:8000"

    def _client(self):
        mc = RadCoPilotClient(self.server_url)
        return mc

    def info(self):
        '''Get information from the RadCoPilot server.'''
        return self._client().info()

    def getAnswer(self, inputText, volumePath):
        '''Get an answer from the RadCoPilot server for the given input text.'''
        return self._client().getAnswer(inputText, volumePath)


class RadCoPilotTest(ScriptedLoadableModuleTest):
    '''RadCoPilot Test class.'''
    def setUp(self):
        '''Set up the scene.'''
        slicer.mrmlScene.Clear()

    def runTest(self):
        '''Run the test.'''
        self.setUp()
        self.test_RadCoPilot1()

    def test_RadCoPilot1(self):
        '''Run the first RadCoPilot test.'''
        self.delayDisplay("Test passed")
