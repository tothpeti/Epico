import os
import glob
import shutil
import pandas as pd
import concurrent.futures


def get_all_datasets_names(path: str,
                           is_prediction_dataset: bool) -> list:
    os.chdir(path)

    tmp = [file for file in glob.glob("*.csv")]

    if is_prediction_dataset:
        tmp.sort(key=sort_csv_files_by_id)
    else:
        tmp.sort(key=sort_csv_files_by_name)

    return tmp


def sort_csv_files_by_id(file: str) -> int:
    return int(file.split('.')[0])


def sort_csv_files_by_name(file: str) -> str:
    return file.split('.')[0]


def get_length_of_test_dataset(path_to_prediction: str,
                               dataset_name: str) -> int:
    return len(pd.read_csv(os.path.join(path_to_prediction, dataset_name)))


def read_all_folders_files_in_memory(directories: list,
                                     path_to_datasets: str,
                                     is_prediction: bool = False) -> list:
    all_datasets = []
    for cur_dir in directories:
        path_to_cur_dir = os.path.join(path_to_datasets, cur_dir)
        tmp_datasets_names = get_all_datasets_names(path_to_cur_dir, is_prediction)
        tmp_datasets = read_all_datasets_in_memory(tmp_datasets_names, path_to_cur_dir)
        all_datasets.append(tmp_datasets)

    return all_datasets


def read_all_datasets_in_memory(datasets_names_list: list,
                                path_to_datasets: str) -> list:
    os.chdir(path_to_datasets)
    tmp = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(pd.read_csv, datasets_names_list)

        for i in results:
            tmp.append(i)

    return tmp


def read_datasets(path_to_datasets: str) -> pd.DataFrame:
    os.chdir(path_to_datasets)

    # Create list of all csv files
    globbed_datasets = glob.glob("*.csv")

    tmp = []
    for csv in globbed_datasets:
        frame = pd.read_csv(csv)
        frame["filename"] = os.path.basename(csv)
        tmp.append(frame)

    return pd.concat(tmp, ignore_index=True)


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
