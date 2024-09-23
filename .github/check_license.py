import os

# Define the license header you are looking for
license_header_first = ["Copyright (c) MONAI Consortium", "limitations under the License."]

# List of file extensions to check
file_extensions = [".py", ".sh", ".ipynb", ".slurm"]


def check_license_in_file(file_path):
    """Check if the file contains the license header"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        for license_header in license_header_first:
            if license_header not in content:
                return False
        return True


def check_license_in_directory(directory):
    """Check for missing license headers in all files in the directory"""
    files_without_license = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                if not check_license_in_file(file_path):
                    files_without_license.append(file_path)

    return files_without_license


if __name__ == "__main__":
    # Change this to the directory you want to check
    directory_to_check = "."

    missing_license_files = check_license_in_directory(directory_to_check)

    if missing_license_files:
        raise FileNotFoundError(f"Copyright is missing in the following files:\n {'\n'.join(missing_license_files)}")
