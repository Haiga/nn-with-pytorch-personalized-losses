import os
import sys

import numpy as np
import torch
import torch.optim as optim
from attr import asdict

from architeture.multiLayer import make_model
from config import Config
from losses import *
from losses.lambdaL import lambdaLoss
from losses.listnet import listnetLoss
from losses.riskLosses.riskFunctions import geoRisk
from utils.dataset import get_data, svmDataset, get_baseline_data
from utils.metrics import mNdcg

torch.manual_seed(2020)
np.random.seed(2020)

if __name__ == '__main__':

    dataset_name = sys.argv[1]
    fold = sys.argv[2]
    epochs = int(sys.argv[3])
    batch_size_queries = int(sys.argv[4])
    name_loss = sys.argv[5]

    out_path = sys.argv[6]
    out_name = sys.argv[7]

    if not os.path.isdir(out_path + "/logs/"):
        os.makedirs(out_path + "/logs/")

    alpha = int(sys.argv[8])
    normalization = int(sys.argv[9])

    k_validation = int(sys.argv[10])
    optimization = sys.argv[11]
    lr = float(sys.argv[12])
    weight_decay = float(sys.argv[13])

    data_infos = svmDataset(dataset_name)
    data_infos.train_data_path = dataset_name + f"/Fold{fold}/Norm.train.txt"
    data_infos.test_data_path = dataset_name + f"/Fold{fold}/Norm.test.txt"
    data_infos.vali_data_path = dataset_name + f"/Fold{fold}/Norm.vali.txt"
    data_infos.baseline_train_data_path = dataset_name + f"/Fold{fold}/baseline.Norm.train.txt"
    data_infos.baseline_test_data_path = dataset_name + f"/Fold{fold}/baseline.Norm.test.txt"
    data_infos.baseline_vali_data_path = dataset_name + f"/Fold{fold}/baseline.Norm.vali.txt"

    # Get data train - to train
    X_train, y_train = get_data(dataset_name, "train")
    y_baseline_train = get_baseline_data(dataset_name, "train")
    N_queries_train = len(X_train)
    assert len(y_train) == len(y_baseline_train)

    # Get data vali - to adjust
    X_vali, y_vali = get_data(dataset_name, "vali")
    y_baseline_vali = get_baseline_data(dataset_name, "vali")
    assert len(y_vali) == len(y_baseline_vali)

    # Get data test - to predict
    X_test, y_test = get_data(dataset_name, "test")
    # y_baseline_test = get_baseline_data(dataset_name, "test")

    X_train = torch.tensor(X_train, requires_grad=True)
    y_train = torch.tensor(y_train, requires_grad=True)
    y_baseline_train = torch.tensor(y_baseline_train, requires_grad=True)

    X_vali = torch.tensor(X_vali, requires_grad=False)
    y_vali = torch.tensor(y_vali, requires_grad=False)
    y_baseline_vali = torch.tensor(y_baseline_vali, requires_grad=False)

    X_test = torch.tensor(X_test, requires_grad=False)
    y_test = torch.tensor(y_test, requires_grad=False)

    N_features = data_infos.num_features

    # net = DoubleLayerNet(N_features)
    config = Config.from_json("config.json")
    net = make_model(n_features=N_features, **asdict(config.model, recurse=False))

    if optimization == "Adam":
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimization == "SGD":
        opt = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_ndcg_score = -1

    with open(out_path + "/logs/" + out_name) as log_out:

        for epoch in range(epochs):
            idx = torch.randperm(N_queries_train)

            X_train = X_train[idx]
            y_train = y_train[idx]

            y_baseline_train = y_baseline_train[idx]

            cur_batch = 0
            for it in range(N_queries_train // batch_size_queries):
                # print(f"started {it}")
                batch_X = X_train[cur_batch: cur_batch + batch_size_queries]
                batch_ys = y_train[cur_batch: cur_batch + batch_size_queries]
                batch_ys_baseline = y_baseline_train[cur_batch: cur_batch + batch_size_queries]
                cur_batch += batch_size_queries

                if len(batch_X) > 0:
                    batch_preds = net(batch_X, None, None)

                    batch_ys = torch.squeeze(batch_ys)
                    batch_ys_baseline = torch.squeeze(batch_ys_baseline)
                    batch_preds = torch.squeeze(batch_preds)

                    if name_loss == "lambdaLoss":
                        batch_loss = lambdaLoss(batch_preds, batch_ys, weighing_scheme="ndcgLoss2PP_scheme")
                    elif name_loss == "listnetLoss":
                        batch_loss = listnetLoss(batch_ys, batch_preds)
                    elif name_loss == "geoRiskListnetLoss":
                        batch_loss = riskLosses.geoRiskListnetLoss(batch_ys, batch_preds,
                                                                   torch.mean(batch_ys_baseline, dim=2), alpha=alpha,
                                                                   normalization=normalization)
                    elif name_loss == "geoRiskLambdaLoss":
                        batch_loss = riskLosses.geoRiskLambdaLoss(batch_ys, batch_preds,
                                                                  torch.mean(batch_ys_baseline, dim=2), alpha=alpha,
                                                                  normalization=normalization)
                    elif name_loss == "zRiskListnetLoss":
                        batch_loss = riskLosses.zRiskListnetLoss(batch_ys, batch_preds,
                                                                 torch.mean(batch_ys_baseline, dim=2), alpha=alpha,
                                                                 normalization=normalization)
                    elif name_loss == "zRiskLambdaLoss":
                        batch_loss = riskLosses.zRiskLambdaLoss(batch_ys, batch_preds,
                                                                torch.mean(batch_ys_baseline, dim=2), alpha=alpha,
                                                                normalization=normalization)
                    elif name_loss == "tRiskListnetLoss":
                        batch_loss = riskLosses.tRiskListnetLoss(batch_ys, batch_preds,
                                                                 torch.mean(batch_ys_baseline, dim=2), alpha=alpha,
                                                                 normalization=normalization)
                    elif name_loss == "tRiskLambdaLoss":
                        batch_loss = riskLosses.tRiskLambdaLoss(batch_ys, batch_preds,
                                                                torch.mean(batch_ys_baseline, dim=2), alpha=alpha,
                                                                normalization=normalization)

                    opt.zero_grad()
                    batch_loss.backward(retain_graph=True)
                    opt.step()
                # print(f"ended {it}")
            with torch.no_grad():
                valid_pred = net(X_vali, None, None)
                ndcg_score = mNdcg(y_vali.numpy(), torch.squeeze(valid_pred).numpy(), k=k_validation)
                ndcg_score_mean = np.mean(ndcg_score)
                if ndcg_score_mean > best_ndcg_score:
                    best_ndcg_score = ndcg_score_mean
                    log = f"#Log -{epoch + 1}- new best ndcg_score_mean - writing predictions"
                    print(log)
                    log_out.write(log + "\n")
                    test_pred = net(X_test, None, None)
                    test_pred_numpy = test_pred.numpy()
                    with open(out_path + "/" + out_name, 'w') as fo:
                        for i in test_pred_numpy:
                            fo.write(f"{i}\n")

                mat = [torch.tensor(ndcg_score)]
                for i in range(y_baseline_vali.shape[2]):
                    ndcg_score_baseline_i = mNdcg(y_vali.numpy(), y_baseline_vali[:, :, i], k=k_validation)
                    mat.append(torch.tensor(ndcg_score_baseline_i))
                mat = torch.stack(mat).t()
                georisk_score = geoRisk(mat, alpha, requires_grad=False)
                georisk_score = georisk_score.numpy()[0]

                log = f"epoch: {epoch + 1} - ndcg@{k_validation}: {ndcg_score_mean:.4f} - georisk-{alpha}: {georisk_score:.4f}"
                print(log)
                log_out.write(log + "\n")
