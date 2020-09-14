import os
import glob
import shutil
import pandas as pd
import concurrent.futures
import numpy as np


def get_all_datasets_names(path, is_prediction):
    os.chdir(path)

    tmp = [file for file in glob.glob("*.csv")]

    if is_prediction:
        tmp.sort(key=sort_csv_files_by_id)
    else:
        tmp.sort(key=sort_csv_files_by_name)

    return tmp


def sort_csv_files_by_id(file):
    return int(file.split('.')[0])


def sort_csv_files_by_name(file):
    return file.split('.')[0]


def get_length_of_test_dataset(path_to_prediction, dataset_name):
    return len(pd.read_csv(os.path.join(path_to_prediction, dataset_name)))


def read_all_folders_files_in_memory(directories, path_to_datasets, is_prediction=False):
    all_datasets = []
    for cur_dir in directories:
        path_to_cur_dir = os.path.join(path_to_datasets, cur_dir)
        tmp_datasets_names = get_all_datasets_names(path_to_cur_dir, is_prediction)
        tmp_datasets = read_all_datasets_in_memory(tmp_datasets_names, path_to_cur_dir)
        all_datasets.append(tmp_datasets)

    return all_datasets


def read_all_datasets_in_memory(datasets_names_list, path_to_datasets):
    os.chdir(path_to_datasets)
    tmp = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(pd.read_csv, datasets_names_list)

        for i in results:
            tmp.append(i)

    return tmp


def put_column_excluded_files_into_folders(path_to_folder):
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

