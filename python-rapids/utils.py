import os
import glob
import shutil

import cudf


def read_datasets(path_to_datasets: str) -> cudf.DataFrame:
    os.chdir(path_to_datasets)

    # Create list of all csv files
    globbed_datasets = glob.glob("*.csv")

    tmp = []
    for csv in globbed_datasets:
        frame = cudf.read_csv(csv)
        frame["filename"] = os.path.basename(csv)
        tmp.append(frame)

    return cudf.concat(tmp, ignore_index=True)


def put_column_excluded_files_into_folders(path_to_folder: str) -> None:
    os.chdir(path_to_folder)

    for f in glob.glob("*.csv"):
        excluded_col_idx = f.split('.')[0].split('_')[-1]
        stripped_file_name = f.split('.')[0].split('_')[0] + '.csv'

        if not os.path.exists(excluded_col_idx):
            os.mkdir(excluded_col_idx)
            os.rename(f, stripped_file_name)
            shutil.move(stripped_file_name, excluded_col_idx)
        else:
            os.rename(f, stripped_file_name)
            shutil.move(stripped_file_name, excluded_col_idx)