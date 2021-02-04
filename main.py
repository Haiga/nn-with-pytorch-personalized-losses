from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# torch.seed = 1000
from losses.approxNDCG import approxNDCGLoss
from utils.dataset import get_data, svmDataset
from losses.lambdaLoss import lambdaLoss


class Net(nn.Module):
    def __init__(self, N_features):
        super(Net, self).__init__()
        self.l1 = nn.Linear(N_features, 10)
        self.l2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = self.l2(x)
        return x


def listnet_loss(y_true, y_predicted):
    """
    y_i: (n_i, 1)
    z_i: (n_i, 1)
    """

    P_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    P_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    return - torch.sum(P_y_true * torch.log(P_y_predicted))

def clean_ndcg_loss(y_true, y_predicted):

    y_true = F.softmax(y_true, dim=1)
    y_predicted = F.softmax(y_predicted, dim=1)
    y_predicted= torch.log(y_predicted)

    corrected_order_y_true = torch.argsort(y_true, descending=True, dim=1)
    ordered_scores_y_predicted = []
    # ordered_scores_y_true = []

    for i in range(y_predicted.shape[0]):
        ordered_scores_y_predicted.append(torch.squeeze(y_predicted[i][corrected_order_y_true[i]]))
        # ordered_scores_y_true.append(torch.squeeze(y_true[i][corrected_order_y_true[i]]))

    ordered_scores_y_predicted = torch.stack(ordered_scores_y_predicted)
    # ordered_scores_y_true = torch.stack(ordered_scores_y_true)

    # inverted_gains = torch.log(torch.arange(1, y_predicted.shape[1] + 1, dtype=torch.float, requires_grad=True)) + 1
    # decrescing_gains = torch.log(torch.arange(-y_predicted.shape[1], y_predicted.shape[1], 2, dtype=torch.float, requires_grad=True))

    decrescing_gains = torch.arange(-y_predicted.shape[1], y_predicted.shape[1], 2, dtype=torch.float,
                  requires_grad=True)
    # decrescing_gains= decrescing_gains ** 3
    dgc = decrescing_gains * ordered_scores_y_predicted

    # dgc = F.softmax(dgc, dim=1)
    # y_predicted = F.softmax(y_predicted, dim=1)

    # return torch.mean(torch.sum(dgc, dim=1))
    return -torch.mean(torch.sum(dgc, dim=1))
    # return torch.var(torch.sum(dgc, dim=1))

# def ndcg_loss(y_true, y_predicted):
#
#     y_true = F.softmax(y_true, dim=1)
#     y_predicted = F.softmax(y_predicted, dim=1)
#
#     corrected_order_y_pred = torch.argsort(y_predicted, descending=True, dim=1)
#     ordered_scores_y_predicted = []
#
#     for i in range(y_predicted.shape[0]):
#         ordered_scores_y_predicted.append(torch.squeeze(y_true[i][corrected_order_y_pred[i]]))
#
#     ordered_scores_y_predicted = torch.stack(ordered_scores_y_predicted)
#
#     inverted_gains = torch.log(torch.arange(1, y_predicted.shape[1] + 1, dtype=torch.float, requires_grad=True)) + 1
#     dcg = torch.sum(ordered_scores_y_predicted / inverted_gains, dim=1)
#
#     corrected_order = torch.argsort(y_true, descending=True, dim=1)
#     ordered_scores_y_true = []
#     # TODO evitar loops e stack
#     for i in range(y_predicted.shape[0]):
#         # ordered_scores_y_predicted.append(torch.squeeze(y_predicted[i][corrected_order_y_pred[i]]))
#         ordered_scores_y_true.append(torch.squeeze(y_true[i][corrected_order[i]]))
#
#     ordered_scores_y_true = torch.stack(ordered_scores_y_true)
#
#     idcg = torch.sum(ordered_scores_y_true / inverted_gains, dim=1)
#
#     # return torch.mean(-(dcg / idcg))
#     return torch.mean(-(dcg / idcg))
#     # return -(dcg/idcg)
# def ndcg_loss(y_true, y_predicted):
#     """
#     y_i: (n_i, 1)
#     z_i: (n_i, 1)
#     """
#
#     y_true = F.softmax(y_true, dim=1)
#     y_predicted = F.softmax(y_predicted, dim=1)
#
#     corrected_order = torch.argsort(y_true, descending=True, dim=1)
#     ordered_scores_y_predicted = []
#     ordered_scores_y_true = []
#     # TODO evitar loops e stack
#     for i in range(y_predicted.shape[0]):
#         ordered_scores_y_predicted.append(torch.squeeze(y_predicted[i][corrected_order[i]]))
#         ordered_scores_y_true.append(torch.squeeze(y_true[i][corrected_order[i]]))
#     ordered_scores_y_predicted = torch.stack(ordered_scores_y_predicted)
#     ordered_scores_y_true = torch.stack(ordered_scores_y_true)
#     inverted_gains = torch.log(torch.arange(1, y_predicted.shape[1] + 1, dtype=torch.float, requires_grad=True)) + 1
#     dcg = torch.sum(ordered_scores_y_predicted / inverted_gains, dim=1)
#     idcg = torch.sum(ordered_scores_y_true / inverted_gains, dim=1)
#
#     # return torch.mean(-(dcg / idcg))
#     return torch.mean((dcg / idcg))
#     # return -(dcg/idcg)


def make_dataset(N_queries_train, N_queries_valid, N_docs_per_query, N_features):
    X, y = get_data("2003_td_dataset", "test")

    X_train = torch.randn(N_queries_train, N_docs_per_query, N_features, requires_grad=True)
    X_valid = torch.randn(N_queries_valid, N_docs_per_query, N_features, requires_grad=True)

    bins = [-2, -1, 0, 1]  # 5 relevances
    ys_final_train = []
    ys_final_valid = []

    ws = torch.randn(N_features, 1)

    # TODO evitar loops e stack
    for i in range(N_queries_train):

        ys_train_score = torch.mm(X_train[i], ws)
        if i < N_queries_valid:
            ys_valid_score = torch.mm(X_valid[i], ws)

        ys_train_rel = torch.Tensor(
            np.digitize(ys_train_score.clone().detach().numpy(), bins=bins)
        )

        if i < N_queries_valid:
            ys_valid_rel = torch.Tensor(
                np.digitize(ys_valid_score.clone().detach().numpy(), bins=bins)
            )

        ys_final_train.append(ys_train_rel)
        if i < N_queries_valid:
            ys_final_valid.append(ys_valid_rel)

    ys_final_train = torch.stack(ys_final_train)
    ys_final_valid = torch.stack(ys_final_valid)

    return X_train, X_valid, ys_final_train, ys_final_valid


def swapped_pairs(ys_pred, ys_target):
    N = ys_target.shape[1]
    Q = ys_target.shape[0]
    swapped = 0
    for q in range(Q):
        for i in range(N - 1):
            for j in range(i + 1, N):
                if ys_target[q][i] < ys_target[q][j]:
                    if ys_pred[q][i] > ys_pred[q][j]:
                        swapped += 1
                elif ys_target[q][i] > ys_target[q][j]:
                    if ys_pred[q][i] < ys_pred[q][j]:
                        swapped += 1
        break
    return swapped


# TODO vetorizar
def ndcg(ys_true, ys_pred):
    def dcg(ys_true, ys_pred):
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort]
        ret = 0
        for i, l in enumerate(ys_true_sorted, 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret

    r = []
    for q in range(ys_true.shape[0]):
        ideal_dcg = dcg(ys_true[q], ys_true[q])
        pred_dcg = dcg(ys_true[q], ys_pred[q])
        r.append(pred_dcg / ideal_dcg)
    return np.mean(r)


if __name__ == '__main__':

    dataset_name = "2003_td_dataset"
    data_infos = svmDataset(dataset_name)

    X_train, y_train = get_data(dataset_name, "test")
    X_vali, y_vali = get_data(dataset_name, "vali")

    X_train = torch.tensor(X_train, requires_grad=True)
    y_train = torch.tensor(y_train, requires_grad=True)
    X_vali = torch.tensor(X_vali, requires_grad=True)
    y_vali = torch.tensor(y_vali, requires_grad=True)

    # N_queries = 100
    N_queries_train = 10
    N_queries_valid = 10
    N_docs_per_query = 1000
    N_features = data_infos.num_features
    epochs = 30
    # batch_size_docs = 20
    batch_size_queries = 3

    # X_train, X_valid, ys_train, ys_valid = make_dataset(N_queries_train, N_queries_valid, N_docs_per_query, N_features)

    net = Net(N_features)
    # opt = optim.Adam(net.parameters(), lr=0.001)
    opt = optim.Adam(net.parameters())
    # opt = optim.SGD(net.parameters(), lr=0.001)
    # opt = optim.SGD(net.parameters(), lr=0.0001)

    for epoch in range(epochs):
        idx = torch.randperm(N_queries_train)

        X_train = X_train[idx]
        y_train = y_train[idx]

        cur_batch = 0
        for it in range(N_queries_train // batch_size_queries):
            # print(f"started {it}")
            batch_X = X_train[cur_batch: cur_batch + batch_size_queries]
            batch_ys = y_train[cur_batch: cur_batch + batch_size_queries]
            cur_batch += batch_size_queries

            opt.zero_grad()
            if len(batch_X) > 0:
                # batch_preds = []
                # for it_querie in range(batch_size_queries):
                #     batch_pred = net(batch_X[it_querie])
                #     batch_preds.append(batch_pred)
                # batch_preds = torch.stack(batch_preds)
                batch_preds = net(batch_X)


                ####
                batch_ys = torch.squeeze(batch_ys)
                batch_preds = torch.squeeze(batch_preds)
                ##
                # batch_loss = lambdaLoss(batch_preds, batch_ys)
                # batch_loss = ndcg_loss(batch_ys, batch_preds)
                # batch_loss = approxNDCGLoss(batch_ys, batch_preds)
                batch_loss = clean_ndcg_loss(batch_ys, batch_preds)
                # batch_loss = listnet_loss(batch_ys, batch_preds)
                batch_loss.backward(retain_graph=True)
                opt.step()
                # batch_loss = ndcg_loss(batch_ys, batch_pred)
                # batch_loss = listnet_loss(batch_ys, batch_pred)
                # batch_loss.backward(retain_graph=True)
                # opt.step()
                # #####
            # print(f"ended {it}")
        with torch.no_grad():
            valid_pred = net(X_vali)
            # valid_swapped_pairs = swapped_pairs(valid_pred, y_vali)##############TODO demora muito a fazer essa contagem
            ndcg_score = ndcg(y_vali, valid_pred).item()
            # ndcg_score = ndcg(y_vali, -1*valid_pred).item()
            # print(f"epoch: {epoch + 1} valid swapped pairs: {valid_swapped_pairs}/{1 * N_docs_per_query * (N_docs_per_query - 1) // 2} ndcg: {ndcg_score:.4f}")
            print(f"epoch: {epoch + 1} - ndcg: {ndcg_score:.4f}")
