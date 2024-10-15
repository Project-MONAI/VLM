import sys
import unittest
from dotenv import load_dotenv
import tempfile
load_dotenv()

sys.path.append("demo/experts")

from expert_torchxrayvision import ExpertTXRV
from expert_monai_vista3d import ExpertVista3D

VISTA_URL = "https://developer.download.nvidia.com/assets/Clara/monai/samples/liver_0.nii.gz"
CXR_URL = "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_ce3d3d98-bf5170fa-8e962da1-97422442-6653c48a_v1.jpg"

class TestVista3D(unittest.TestCase):
    # def test_run_vista3d(self):
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         prompt = "This seems a CT image. Let me trigger <VISTA3D(everything)>."
    #         vista3d = ExpertVista3D()
    #         self.assertTrue(vista3d.is_mentioned(prompt))

    #         output = vista3d.run(image_url=VISTA_URL, input=prompt, output_dir=temp_dir)
    #         print(output)
    #         self.assertTrue(output is not None)


    def test_run_cxr(self):
        input = "This seems a CXR image. Let me trigger <CXR>."
        cxr = ExpertTXRV()
        self.assertTrue(cxr.is_mentioned(input))

        output_text, file, _ = cxr.run(image_url=CXR_URL, prompt="")
        print(output_text)
        self.assertTrue(output_text is not None)
        self.assertTrue(file is None)

if __name__ == "__main__":
    unittest.main()
