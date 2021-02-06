import math

import numpy as np
import torch
from scipy.stats import norm


def getGeoRiskDefault(mat, alpha):
    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = np.array([0.0] * numQueries)
    Si = np.array([0.0] * numSystems)
    geoRisk = np.array([0.0] * numSystems)
    zRisk = np.array([0.0] * numSystems)
    mSi = np.array([0.0] * numSystems)

    for i in range(numSystems):
        Si[i] = np.sum(mat[:, i])
        mSi[i] = np.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j] = np.sum(mat[j, :])

    N = np.sum(Tj)

    for i in range(numSystems):
        tempZRisk = 0
        for j in range(numQueries):
            eij = Si[i] * (Tj[j] / N)
            xij_eij = mat[j, i] - eij
            if eij != 0:
                ziq = xij_eij / math.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq = (1 + alpha) * ziq
            tempZRisk = tempZRisk + ziq
        zRisk[i] = tempZRisk

    c = numQueries
    for i in range(numSystems):
        ncd = norm.cdf(zRisk[i] / c)
        geoRisk[i] = math.sqrt((Si[i] / c) * ncd)

    return geoRisk


def dcg(true_relevance, pred_relevance, k=5, gains='linear', use_numpy=False):
    np_true_relevance = np.array(true_relevance)
    if not use_numpy:
        args_pred = [i[0] for i in sorted(enumerate(pred_relevance), key=lambda p: p[1], reverse=True)]
    else:
        args_pred = np.argsort(pred_relevance)[::-1]
    if np_true_relevance.shape[0] < k:
        k = np_true_relevance.shape[0]
    order_true_relevance = np.take(np_true_relevance, args_pred[:k])

    if gains == "exponential":
        gains = 2 ** order_true_relevance - 1
    elif gains == "linear":
        gains = order_true_relevance
    else:
        raise ValueError("Invalid gains option.")

    discounts = np.log2(np.arange(k) + 2)
    return np.sum(gains / discounts)


def ndcg(true_relevance, pred_relevance, k=5, no_relevant=True, gains='linear', use_numpy=False):
    dcg_atk = dcg(true_relevance, pred_relevance, k, gains, use_numpy)
    idcg_atk = dcg(true_relevance, true_relevance, k, gains, use_numpy)
    if idcg_atk == 0 and no_relevant: return 1.0
    if idcg_atk == 0 and not no_relevant: return 0.0
    return dcg_atk / idcg_atk


def mNdcg(true_relevance, pred_relevance, k=5, no_relevant=True, gains='linear', use_numpy=False):
    np_true_relevance = np.array(true_relevance)
    num_queries = np_true_relevance.shape[0]
    return [ndcg(true_relevance[i], pred_relevance[i], k, no_relevant, gains, use_numpy) for i in range(num_queries)]


def torchNdcg(ys_true, ys_pred, k=None, return_type='list'):
    def dcg(ys_true, ys_pred, k=None):
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort]
        ret = 0
        if not k is None:
            ys_true_sorted = ys_true_sorted[:k]
        for i, l in enumerate(ys_true_sorted, 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret

    r = []
    for q in range(ys_true.shape[0]):
        ideal_dcg = dcg(ys_true[q], ys_true[q], k=k)
        pred_dcg = dcg(ys_true[q], ys_pred[q], k=k)
        r.append(pred_dcg / ideal_dcg)
    if return_type == 'tensor':
        return torch.tensor(r)
    return r


if __name__ == "__main__":
    predicted_relevance = np.asarray([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    true_relevance = np.asarray([[0, 0, 0, 0, 1], [10, 0, 0, 0, 0]])

    r = mNdcg(true_relevance, predicted_relevance, k=5, no_relevant=True, gains='linear', use_numpy=False)
    print(r)
    print(np.mean(r))
    r = mNdcg(true_relevance, predicted_relevance, k=5, no_relevant=True, gains='linear', use_numpy=True)
    print(r)
    print(np.mean(r))
