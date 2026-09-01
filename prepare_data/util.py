
import shutil
import re
import os


def copy_json_sidecar(source_file, dest_file):
    # check if source_file has a json sidecar
    json_file = re.sub(r'(\.nii(\.gz)?)$', r'.json', source_file)
    if os.path.exists(json_file):
        dest_json_file = re.sub(r'(\.nii(\.gz)?)$', r'.json', dest_file)
        shutil.copy2(json_file, dest_json_file)