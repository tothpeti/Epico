import os
import glob
import shutil


def get_all_datasets(path):
    os.chdir(path)
    return [file for file in glob.glob("*.csv")]


def put_column_excluded_files_into_folders(path_to_folder):
    os.chdir(path_to_folder)

    for f in glob.glob("*.csv"):
        excluded_col_idx = f.split('.')[0].split('_')[-1]

        if not os.path.exists(excluded_col_idx):
            os.mkdir(excluded_col_idx)
            shutil.move(f, excluded_col_idx)
        else:
            shutil.move(f, excluded_col_idx)

