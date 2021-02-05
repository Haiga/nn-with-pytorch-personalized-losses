from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, ndcg_score
import numpy as np
from metrics import mNdcg
from utils.dataset import store_baseline_data, svmDataset

mem = Memory("./mycache")


@mem.cache
def get_data(name_file):
    data = load_svmlight_file(name_file, query_id=True)
    return data[0], data[1], data[2]


path = "D:\\Colecoes\\"
# name_dataset = "2003_td_dataset-norm"
name_dataset = "web10k-norm"
for fold in range(1, 5 + 1):
    for t in ["train", "test", "vali"]:
        X_train, y_train, queries_id = get_data(path + "/" + name_dataset + f"/Fold{fold}/Norm." + "train" + ".txt")
        for m in ["LinearRegression", "DecisionTreeRegressor"]:

            if m == "LinearRegression":
                model = LinearRegression()
            elif m == "RandomForestRegressor":
                model = RandomForestRegressor(n_estimators=2, n_jobs=-1, verbose=1)
            elif m == "DecisionTreeRegressor":
                model = DecisionTreeRegressor(max_depth=10)
            model.fit(X_train, y_train)

            if t != "train":
                X, y, queries_id = get_data(path + "/" + name_dataset + f"/Fold{fold}/Norm." + t + ".txt")
            else:
                X = X_train
                y = y_train
            p = model.predict(X)

            rmse = mean_squared_error(y, p)

            queries_id = np.array(queries_id)
            queries_id_unique = []
            ant = queries_id[0]
            all_preds = []
            all_trues = []
            true_relevance = []
            pred_relevance = []
            # start = True
            conts = []
            for i in range(queries_id.size):
                if ant != queries_id[i]:
                    conts.append(len(pred_relevance))
                    all_preds.append(pred_relevance)
                    all_trues.append(true_relevance)
                    true_relevance = []
                    pred_relevance = []
                    queries_id_unique.append(ant)

                pred_relevance.append(p[i])
                true_relevance.append(y[i])
                ant = queries_id[i]

            conts.append(len(pred_relevance))
            all_preds.append(pred_relevance)
            all_trues.append(true_relevance)
            queries_id_unique.append(ant)

            x = mNdcg(all_trues, all_preds, k=10, no_relevant=True, gains='exponential', use_numpy=True)
            info_dataset = svmDataset(name_dataset)
            info_dataset.baseline_train_data_path = path + "/" + name_dataset + f"/Fold{fold}/baseline.Norm." + "train" + ".txt"
            info_dataset.baseline_test_data_path = path + "/" + name_dataset + f"/Fold{fold}/baseline.Norm." + "test" + ".txt"
            info_dataset.baseline_vali_data_path = path + "/" + name_dataset + f"/Fold{fold}/baseline.Norm." + "vali" + ".txt"
            store_baseline_data(m, info_dataset, queries_id, p, type_file=t)
            mNdcg_mean = np.mean(x)
            print("Trained: " + m + " - " + path + "/" + name_dataset + f"/Fold{fold}/Norm." + t + ".txt")
            print(f"NDCG@{10}: {mNdcg_mean}")
            print(f"RMSE: {rmse}")
            print(f"-----------------------")
