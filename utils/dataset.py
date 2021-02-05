import numpy as np
import os
from sklearn.datasets import load_svmlight_file


class svmDataset():
    def __init__(self, dataset_name):
        self.train_data_path = "D:\\Colecoes\\2003_td_dataset\\Fold1\\train.txt"
        self.test_data_path = "D:\\Colecoes\\2003_td_dataset\\Fold1\\test.txt"
        self.vali_data_path = "D:\\Colecoes\\2003_td_dataset\\Fold1\\vali.txt"

        self.baseline_train_data_path = "D:\\Colecoes\\2003_td_dataset\\Fold1\\baseline.train.txt"
        self.baseline_test_data_path = "D:\\Colecoes\\2003_td_dataset\\Fold1\\baseline.test.txt"
        self.baseline_vali_data_path = "D:\\Colecoes\\2003_td_dataset\\Fold1\\baseline.vali.txt"

        self.docs_per_query = 1000
        self.queries_on_test = 10
        self.queries_on_train = 10

        if dataset_name == "2003_td_dataset":
            self.num_features = 64
            self.normalized_num_docs = True

        elif "web10k" in dataset_name:

            self.num_features = 136
            self.normalized_num_docs = True


def get_data(info_dataset, type_file="train"):
    # info_dataset = svmDataset(dataset_name)

    if type_file == "train":
        data = load_svmlight_file(info_dataset.train_data_path, query_id=True)
    elif type_file == "test":
        data = load_svmlight_file(info_dataset.test_data_path, query_id=True)
    elif type_file == "vali":
        data = load_svmlight_file(info_dataset.vali_data_path, query_id=True)

    labels_by_query = []
    features_docs_by_query = []

    queries_ids = np.array(data[2])
    ant = queries_ids[0]

    temp_labels_by_query = []
    temp_features_docs_by_query = []

    # np.array(data[0][0].shape[1], data[0][0].shape[1])
    for i in range(queries_ids.size):
        if ant != queries_ids[i]:
            labels_by_query.append(temp_labels_by_query)
            features_docs_by_query.append(temp_features_docs_by_query)

            temp_labels_by_query = []
            temp_features_docs_by_query = []

        temp_labels_by_query.append(data[1][i])
        temp_features_docs_by_query.append(data[0][i].toarray().reshape(-1).astype(np.float32))
        ant = queries_ids[i]

    labels_by_query.append(temp_labels_by_query)
    features_docs_by_query.append(temp_features_docs_by_query)

    return features_docs_by_query, labels_by_query


def load_baseline_file(name_file):
    with open(name_file, "r") as f:
        data = f.readlines()

    data_formatted = []
    queries_ids = []
    for line in data:
        splitted_line = line.replace("\n", "").split(",")
        queries_ids.append(int(splitted_line[0]))
        data_formatted.append([float(x) for x in splitted_line[1:]])

    data_formatted = np.array(data_formatted, dtype=np.float32)
    queries_ids = np.array(queries_ids)
    ant = queries_ids[0]

    baselines_by_query = []
    temp_baselines_by_query = []

    # np.array(data[0][0].shape[1], data[0][0].shape[1])
    for i in range(queries_ids.size):
        if ant != queries_ids[i]:
            baselines_by_query.append(temp_baselines_by_query)
            temp_baselines_by_query = []

        temp_baselines_by_query.append(data_formatted[i])
        ant = queries_ids[i]

    baselines_by_query.append(temp_baselines_by_query)

    return baselines_by_query


def store_baseline_data(name_method, info_dataset, queries_ids, predicted_values, append=True, type_file="train"):
    assert len(queries_ids) == len(predicted_values)

    # info_dataset = svmDataset(dataset_name)

    if type_file == "train":
        name_file = info_dataset.baseline_train_data_path
    elif type_file == "test":
        name_file = info_dataset.baseline_test_data_path
    elif type_file == "vali":
        name_file = info_dataset.baseline_vali_data_path

    if append:
        # checar se já existe
        if not os.path.isfile(name_file):
            append = False
    if append:
        with open(name_file, "r") as existing_file:
            data = existing_file.readlines()
        with open(name_file.replace("baseline", "baseline.info"), "a") as existing_file:
            existing_file.write(name_method + "\n")

    with open(name_file, "w") as f:
        if append:
            assert len(queries_ids) == len(data)
            for i in range(len(queries_ids)):
                # f.write(f"{queries_ids[i]},{predicted_values[i]}\n")
                current_line = data[i].replace("\n", "")
                f.write(f"{current_line},{round(predicted_values[i], ndigits=5)}\n")

        else:
            for i in range(len(queries_ids)):
                f.write(f"{queries_ids[i]},{round(predicted_values[i], ndigits=5)}\n")
            with open(name_file.replace("baseline", "baseline.info"), "w") as existing_file:
                existing_file.write(name_method + "\n")


def get_baseline_data(info_dataset, type_file="train"):
    # info_dataset = svmDataset(dataset_name)

    if type_file == "train":
        data = load_baseline_file(info_dataset.baseline_train_data_path)
    elif type_file == "test":
        data = load_baseline_file(info_dataset.baseline_test_data_path)
    elif type_file == "vali":
        data = load_baseline_file(info_dataset.baseline_vali_data_path)

    return data
