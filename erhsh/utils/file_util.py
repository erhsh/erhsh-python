import os
from datetime import datetime


def list_sub_dirs(dir_path):
    if not os.path.exists(dir_path):
        raise RuntimeError("invalid dir path.")

    if not os.path.isdir(dir_path):
        raise RuntimeError("invalid dir path.")

    sub_dirs = []
    sub_files = os.listdir(dir_path)
    for sub_file_name in sub_files:
        sub_file_path = os.path.join(dir_path, sub_file_name)
        if not os.path.isdir(sub_file_path):
            continue
        sub_dirs.append(sub_file_path)
    return sub_dirs


def flatten_dir_path(dir_path):
    sub_dirs = list_sub_dirs(dir_path)
    for sub_dir_path in sub_dirs:
        sub_sub_dirs = list_sub_dirs(sub_dir_path)
        for sub_sub_dir_path in sub_sub_dirs:
            sub_sub_dir_name = os.path.basename(sub_sub_dir_path)
            target_dir_path = '_'.join([sub_dir_path, sub_sub_dir_name])
            print(f"move: {sub_sub_dir_path} -> {target_dir_path}")
            os.renames(sub_sub_dir_path, target_dir_path)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return dir_path
    return create_dir(dir_path + "." + datetime.now().strftime("%Y%m%d%H%M%S"))
