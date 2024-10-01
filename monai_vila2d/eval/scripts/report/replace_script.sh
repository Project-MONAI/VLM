#!/bin/bash

# Usage: ./replace_script.sh <package_name> <script_name> <local_script_path>

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <package_name> <script_name> <local_script_path>"
    exit 1
fi

PACKAGE_NAME=$1
SCRIPT_NAME=$2
LOCAL_SCRIPT_PATH=$3

# Find the package location using pip
PACKAGE_PATH=$(pip show "$PACKAGE_NAME" | grep Location | awk '{print $2}')

if [ -z "$PACKAGE_PATH" ]; then
    echo "Package '$PACKAGE_NAME' is not installed."
    exit 1
fi

# Construct the full path to the target script in the package
TARGET_SCRIPT_PATH="$PACKAGE_PATH/$PACKAGE_NAME/$SCRIPT_NAME"

# Check if the target script exists
if [ ! -f "$TARGET_SCRIPT_PATH" ]; then
    echo "Script '$SCRIPT_NAME' not found in package '$PACKAGE_NAME'."
    exit 1
fi

# Replace the target script with the local script
cp "$LOCAL_SCRIPT_PATH" "$TARGET_SCRIPT_PATH"

if [ $? -eq 0 ]; then
    echo "Successfully replaced '$TARGET_SCRIPT_PATH' with '$LOCAL_SCRIPT_PATH'."
else
    echo "Failed to replace the script. Please check the paths and try again."
    exit 1
fi
