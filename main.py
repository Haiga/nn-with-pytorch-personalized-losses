import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# torch.seed = 1000
from losses.approxNDCG import approxNDCGLoss
from utils.dataset import get_data, svmDataset, get_baseline_data
from losses.geoLambdaLoss import geoLambdaLoss
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
    P_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    P_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    return - torch.sum(P_y_true * torch.log(P_y_predicted))


def my_listnet_loss(y_true, y_predicted):
    P_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    P_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    # return - torch.sum(P_y_true * torch.log(P_y_predicted))
    r = F.sigmoid(P_y_true * torch.log(P_y_predicted))
    return -torch.sum(r)
    # return - torch.sum(P_y_true * torch.log(P_y_predicted))
    # return - torch.mean(P_y_true * torch.log(P_y_predicted))


def georisk_listnet_loss(y_true, y_predicted, y_baselines):
    P_y_baselines = F.softmax(y_baselines, dim=1)
    P_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    P_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    # r = F.sigmoid(P_y_true * torch.log(P_y_predicted))
    mat = []
    mat.append(P_y_true * torch.log(P_y_predicted))
    for i in range(P_y_baselines.shape[2]):
        mat.append(P_y_true * torch.log(P_y_baselines)[:, :, i])
    mat = torch.stack(mat)
    mat = torch.sum(mat, dim=2)
    mat = -1 * mat.t()

    ######################
    mat = mat - torch.min(mat)
    mat = mat / torch.max(mat)
    # F.softmax(mat)
    # F.sigmoid(mat)

    alpha = torch.tensor([3], requires_grad=True, dtype=torch.float)
    numQueries = mat.shape[0]
    Si = torch.sum(mat[:, 0])
    Tj = torch.sum(mat, dim=1)
    N = torch.sum(Tj)

    # tempZRisk = 0
    ####
    # eij = Si * (Tj / N)
    xij_eij = mat[:, 0] - Si * (Tj / N)
    den = torch.sqrt(Si * (Tj / N))
    div = xij_eij / den
    less0 = (mat[:, 0] - Si * (Tj / N)) / (torch.sqrt(Si * (Tj / N))) < 0
    less0 = alpha * less0
    zRisk = div * less0 + div
    zRisk = torch.sum(zRisk)
    return -zRisk
    # c = numQueries
    # value = zRisk / c
    # m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    # ncd = m.cdf(value)
    # return torch.sqrt((Si / c) * ncd)
    # return -torch.sqrt((Si / c) * ncd)

def geo(mat, alpha):
    alpha = torch.tensor([alpha], requires_grad=True, dtype=torch.float)
    numQueries = mat.shape[0]
    Si = torch.sum(mat[:, 0])
    Tj = torch.sum(mat, dim=1)
    N = torch.sum(Tj)

    # tempZRisk = 0
    ####
    # eij = Si * (Tj / N)
    xij_eij = mat[:, 0] - Si * (Tj / N)
    den = torch.sqrt(Si * (Tj / N))
    div = xij_eij / den
    less0 = (mat[:, 0] - Si * (Tj / N)) / (torch.sqrt(Si * (Tj / N))) < 0
    less0 = alpha * less0
    zRisk = div * less0 + div
    zRisk = torch.sum(zRisk)
    # ####
    # for j in range(numQueries):
    #     eij = Si * (Tj[j] / N)
    #     xij_eij = mat[j, 0] - eij
    #     if eij != 0:
    #         ziq = xij_eij / torch.sqrt(eij)
    #     else:
    #         ziq = 0
    #     if xij_eij < 0:
    #         ziq = (1 + alpha) * ziq
    #     tempZRisk = tempZRisk + ziq
    # zRisk = tempZRisk

    c = numQueries
    value = zRisk / c
    m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ncd = m.cdf(value)
    # return -torch.sqrt((Si / c) * ncd)
    return -torch.sqrt((Si / c) * ncd)

def clean_ndcg_loss(y_true, y_predicted):
    y_true = F.softmax(y_true, dim=1)
    y_predicted = F.softmax(y_predicted, dim=1)
    y_predicted = torch.log(y_predicted)

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


def ndcg_loss(y_true, y_predicted):
    y_true = F.softmax(y_true, dim=1)
    y_predicted = F.softmax(y_predicted, dim=1)

    corrected_order_y_pred = torch.argsort(y_predicted, descending=True, dim=1)
    ordered_scores_y_predicted = []

    for i in range(y_predicted.shape[0]):
        ordered_scores_y_predicted.append(torch.squeeze(y_true[i][corrected_order_y_pred[i]]))

    ordered_scores_y_predicted = torch.stack(ordered_scores_y_predicted)

    inverted_gains = torch.log(torch.arange(1, y_true.shape[1] + 1, dtype=torch.float, requires_grad=True)) + 1
    dcg = torch.sum(ordered_scores_y_predicted / inverted_gains, dim=1)

    corrected_order = torch.argsort(y_true, descending=True, dim=1)
    ordered_scores_y_true = []
    # TODO evitar loops e stack
    for i in range(y_predicted.shape[0]):
        # ordered_scores_y_predicted.append(torch.squeeze(y_predicted[i][corrected_order_y_pred[i]]))
        ordered_scores_y_true.append(torch.squeeze(y_true[i][corrected_order[i]]))

    ordered_scores_y_true = torch.stack(ordered_scores_y_true)

    # idcg = torch.sum(ordered_scores_y_true / inverted_gains, dim=1)

    # return torch.mean(-(dcg / idcg))
    # return torch.mean(-(dcg / idcg))
    return torch.mean(-(dcg))
    # return -(dcg/idcg)


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
    # i = torch.tensor([[0.0500, 0.1500, 0.3000, 0.4500, 0.5500, 0.4000, 0.3500, 0.3000],
    #               [0.2500, 0.2000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.2500],
    #               [0.2500, 0.2500, 0.2500, 0.2500, 0.4000, 0.1500, 0.4000, 0.1500],
    #               [0.4000, 0.2000, 0.4500, 0.2000, 0.4500, 0.2000, 0.2542, 0.2629],
    #               [0.2802, 0.2975, 0.3061, 0.2918, 0.2994, 0.3147, 0.3301, 0.3378]])
    # u = geo(i, 3)
    # # 0.31438308416523303
    # print(u)
    dataset_name = "2003_td_dataset"
    data_infos = svmDataset(dataset_name)

    X_train, y_train = get_data(dataset_name, "test")
    y_baseline_train = get_baseline_data(dataset_name, "test")

    X_vali, y_vali = get_data(dataset_name, "vali")
    y_baseline_vali = get_baseline_data(dataset_name, "vali")

    X_train = torch.tensor(X_train, requires_grad=True)
    y_train = torch.tensor(y_train, requires_grad=True)
    y_baseline_train = torch.tensor(y_baseline_train, requires_grad=True)

    X_vali = torch.tensor(X_vali, requires_grad=True)
    y_vali = torch.tensor(y_vali, requires_grad=True)
    y_baseline_vali = torch.tensor(y_baseline_vali, requires_grad=True)

    # N_queries = 100
    N_queries_train = 10
    N_queries_valid = 10
    N_docs_per_query = 1000
    N_features = data_infos.num_features
    epochs = 40
    # batch_size_docs = 20
    batch_size_queries = 10

    net = Net(N_features)
    # opt = optim.Adam(net.parameters(), lr=0.01)
    opt = optim.Adam(net.parameters())

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
                batch_preds = net(batch_X)

                batch_ys = torch.squeeze(batch_ys)
                batch_ys_baseline = torch.squeeze(batch_ys_baseline)
                batch_preds = torch.squeeze(batch_preds)

                # batch_loss = lambdaLoss(batch_preds, batch_ys, weighing_scheme="ndcgLoss2PP_scheme")
                # batch_loss = geoLambdaLoss(batch_preds, batch_ys, weighing_scheme="ndcgLoss2PP_scheme")
                # batch_loss = ndcg_loss(batch_ys, batch_preds)
                # batch_loss = approxNDCGLoss(batch_ys, batch_preds)
                # batch_loss = clean_ndcg_loss(batch_ys, batch_preds)
                # batch_loss = listnet_loss(batch_ys, batch_preds)
                batch_loss = georisk_listnet_loss(batch_ys, batch_preds, batch_ys_baseline)
                batch_loss.backward(retain_graph=True)
                opt.step()
            # print(f"ended {it}")
        with torch.no_grad():
            valid_pred = net(X_vali)
            ndcg_score = ndcg(y_vali, valid_pred).item()
            print(f"epoch: {epoch + 1} - ndcg: {ndcg_score:.4f}")
