import numpy as np
import torch
import torch.optim as optim
from attr import asdict
from torch import nn

from architeture.multiLayer import make_model
from architeture.tripleLayer import TripleLayerNet
from config import Config
from losses.lambdaL import lambdaLoss
from losses.listnet import listnetLoss
from losses.ordinal import ordinalLoss
from losses.riskLosses.riskFunctions import geoRisk
from utils.dataset import get_data, svmDataset, get_baseline_data
from architeture.doubleLayer import DoubleLayerNet
import losses
from utils.metrics import torchNdcg, mNdcg

if __name__ == '__main__':

    dataset_name = "web10k-norm"
    # dataset_name = "2003_td_dataset"
    data_infos = svmDataset(dataset_name)

    home = "D:\\Colecoes"
    fold = "1"
    data_infos.train_data_path = home + "/BD/" + dataset_name + f"/Fold{fold}/Norm.train.txt"
    data_infos.test_data_path = home + "/BD/" + dataset_name + f"/Fold{fold}/Norm.test.txt"
    data_infos.vali_data_path = home + "/BD/" + dataset_name + f"/Fold{fold}/Norm.vali.txt"
    data_infos.baseline_train_data_path = home + "/BD/" + dataset_name + f"/Fold{fold}/baseline.Norm.train.txt"
    data_infos.baseline_test_data_path = home + "/BD/" + dataset_name + f"/Fold{fold}/baseline.Norm.test.txt"
    data_infos.baseline_vali_data_path = home + "/BD/" + dataset_name + f"/Fold{fold}/baseline.Norm.vali.txt"

    X_train, y_train = get_data(data_infos, "test")
    y_baseline_train = get_baseline_data(data_infos, "test")

    N_queries_train = len(X_train)
    print(N_queries_train)
    X_vali, y_vali = get_data(data_infos, "vali")
    y_baseline_vali = get_baseline_data(data_infos, "vali")

    X_train = torch.tensor(X_train, requires_grad=True)
    y_train = torch.tensor(y_train, requires_grad=True)
    # y_train = torch.tanh(y_train)
    y_baseline_train = torch.tensor(y_baseline_train, requires_grad=True)

    X_vali = torch.tensor(X_vali, requires_grad=True)
    y_vali = torch.tensor(y_vali, requires_grad=True)
    y_baseline_vali = torch.tensor(y_baseline_vali, requires_grad=True)

    # N_queries = 100
    # N_queries_train = 10
    # N_queries_valid = 10
    # N_docs_per_query = 1000
    N_features = data_infos.num_features
    epochs = 50
    # batch_size_docs = 20
    # batch_size_queries = 10
    batch_size_queries = 200

    # net = Net(N_features)
    net = DoubleLayerNet(N_features)
    # net = TripleLayerNet(N_features)
    # config = Config.from_json("config.json")
    # # config.model.d
    # config.model.fc_model['dropout'] = 0.2
    # net = make_model(n_features=N_features, **asdict(config.model, recurse=False))
    # opt = optim.Adam(net.parameters(), lr=0.01)
    # opt = optim.Adam(net.parameters(), lr=0.1)
    opt = optim.Adam(net.parameters(), lr=0.001)

    # opt = optim.SGD(net.parameters(), lr=0.01)
    # opt = optim.SGD(net.parameters(), lr=1)###########com subtração min no geo
    # opt = optim.SGD(net.parameters(), lr=0.0001)

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

            opt.zero_grad()
            if len(batch_X) > 0:
                batch_preds = net(batch_X, None, None)

                batch_ys = torch.squeeze(batch_ys)
                batch_ys_baseline = torch.squeeze(batch_ys_baseline)
                batch_preds = torch.squeeze(batch_preds)

                # batch_loss = ordinalLoss(batch_preds[0], batch_ys[0], 2)
                # batch_loss = lambdaLoss(batch_preds, batch_ys, weighing_scheme="ndcgLoss2PP_scheme")
                # batch_loss = geoLambdaLoss(batch_preds, batch_ys, weighing_scheme="ndcgLoss2PP_scheme")
                # batch_loss = ndcg_loss(batch_ys, batch_preds)
                # batch_loss = approxNDCGLoss(batch_ys, batch_preds)
                # batch_loss = clean_ndcg_loss(batch_ys, batch_preds)

                # batch_loss = georisk_listnet_loss(batch_ys, batch_preds, batch_ys_baseline)
                # batch_loss = losses.riskLosses.geoRiskListnetLoss(batch_ys, batch_preds, batch_ys_baseline, alpha=2)
                # b = nn.MSELoss()
                # batch_loss = b(batch_preds, batch_ys)
                # batch_loss = losses.riskLosses.tRiskListnetLoss(batch_ys, batch_preds,
                #                                                 torch.mean(batch_ys_baseline, dim=2), alpha=1,
                #                                                 normalization=2)######
                # batch_loss = losses.riskLosses.tRiskListnetLoss(batch_ys, batch_preds,
                #                                                 batch_ys_baseline[:,:,0], alpha=1,
                #                                                 normalization=2)  ######
                # batch_loss = losses.riskLosses.geoRiskListnetLoss(batch_ys, batch_preds,batch_ys_baseline, alpha=1,
                #                                                 normalization=1)

                # batch_loss = losses.riskLosses.tRiskLambdaLoss(batch_ys, batch_preds,
                #                                                 torch.mean(batch_ys_baseline, dim=2), alpha=1,
                #                                                 normalization=0)
                # batch_loss = listnetLoss(batch_ys, batch_preds)
                baselines_p = batch_ys_baseline
                # baselines_p = batch_ys
                # baselines_p = torch.mean(batch_ys_baseline, dim=2)
                # batch_loss = losses.riskLosses.tRiskLambdaLoss(batch_ys, batch_preds,###############
                #                                                 baselines_p,
                #                                                 listnet_transformation=1,
                #                                                 negative=1, alpha=2)
                # batch_loss = losses.riskLosses.zRiskListnetLoss(batch_ys, batch_preds,
                #                                                 y_baselines=None,
                #                                                 listnet_transformation=2,
                #                                                 add_ideal_ranking_to_mat=2, alpha=2, return_strategy=2,
                #                                                 negative=1)
                batch_loss = nn.MSELoss()(batch_preds, batch_ys)
                # batch_loss.backward(retain_graph=True)
                batch_loss.backward(retain_graph=True)
                print(batch_loss)
                opt.step()
            # print(f"ended {it}")
        with torch.no_grad():
            valid_pred = net(X_vali, None, None)
            train_pred = net(X_train[:600], None, None)
            ndcg_score = torchNdcg(y_vali, valid_pred, k=10, return_type='tensor')
            ndcg_score_t = torchNdcg(y_train[:600], train_pred, k=10, return_type='tensor')
            # ndcg_score_train = torchNdcg(y_train, net(X_train, None, None), k=10, return_type='tensor')
            ndcg_score_mean = np.mean(ndcg_score.numpy())
            ndcg_score_mean_t = np.mean(ndcg_score_t.numpy())
            # ndcg_score_mean_train = ndcg_score_train.numpy()[0]

            ##############################
            # mat = [ndcg_score]
            # for i in range(y_baseline_vali.shape[2]):
            #     ndcg_score_baseline_i = torchNdcg(y_vali, y_baseline_vali[:, :, i], k=None, return_type='tensor')
            #     mat.append(ndcg_score_baseline_i)
            # mat = torch.stack(mat).t()
            # georisk_score = geoRisk(mat, 2, requires_grad=False)
            # georisk_score = georisk_score.numpy()[0]
            ###################################
            # print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean:.4f}")

            # valid_pred = net(X_vali)
            # ndcg_score = torchNdcg(y_vali, valid_pred, k=100, return_type='tensor')
            # ndcg_score_mean = ndcg_score.numpy()[0]
            # mat = [ndcg_score]
            # for i in range(y_baseline_vali.shape[2]):
            #     ndcg_score_baseline_i = torchNdcg(y_vali, y_baseline_vali[:, :, i], k=100, return_type='tensor')
            #     mat.append(ndcg_score_baseline_i)
            # mat = torch.stack(mat).t()
            # georisk_score = geoRisk(mat, 2, requires_grad=False, remove_nan=True)
            # georisk_score = georisk_score.numpy()[0]
            # # print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean:.4f}")

            # valid_pred = net(X_vali)
            # ndcg_score = mNdcg(y_vali.numpy(), torch.squeeze(valid_pred).numpy(), k=100)
            # ndcg_score_mean = np.mean(ndcg_score)
            # mat = [torch.tensor(ndcg_score)]
            # for i in range(y_baseline_vali.shape[2]):
            #     ndcg_score_baseline_i = mNdcg(y_vali.numpy(), y_baseline_vali[:, :, i], k=100)
            #     mat.append(torch.tensor(ndcg_score_baseline_i))
            # mat = torch.stack(mat).t()
            # georisk_score = geoRisk(mat, 2, requires_grad=False, remove_nan=True)
            # georisk_score = georisk_score.numpy()[0]
            # # print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean:.4f}")

            # print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean:.4f} - ndcgtrain: {ndcg_score_mean_train:.4f} - georisk: {georisk_score:.4f}")
            # print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean:.4f} - ndcgtrain: {ndcg_score_mean_test:.4f} - georisk: {georisk_score:.4f}")
            # print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean:.4f} - georisk: {georisk_score:.4f}")
            print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean:.4f} ")
            print(f"epoch: {epoch + 1} - ndcg: {ndcg_score_mean_t:.4f} ")
            print(f"---")
