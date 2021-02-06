import re
import sys

import numpy as np

from utils.dataset import load_baseline_file
from utils.metrics import mNdcg, getGeoRiskDefault


def read_queries_id_true_relevance(name_file):
    with open(name_file, 'r') as f:
        lines = f.readlines()

    queries_arr = []
    relevance_arr = []
    for line in lines:
        r = re.match("([0-9]*) qid:([0-9]*)", line)
        query_id = int(r.groups()[1])
        relevance = int(r.groups()[0])

        queries_arr.append(query_id)
        relevance_arr.append(relevance)

    return queries_arr, relevance_arr


def read_predicted(name_file):
    with open(name_file, 'r') as f:
        lines = f.readlines()

    preds = []
    for line in lines:
        r = float(line.replace("\n", ""))
        preds.append(r)
    return preds


def group_queries(queries_id, y, p):
    queries_id = np.array(queries_id)
    all_preds = []
    all_trues = []
    true_relevance = []
    pred_relevance = []
    ant = queries_id[0]

    for i in range(queries_id.size):
        if ant != queries_id[i]:
            all_preds.append(pred_relevance)
            all_trues.append(true_relevance)
            true_relevance = []
            pred_relevance = []

        pred_relevance.append(p[i])
        true_relevance.append(y[i])
        ant = queries_id[i]

    all_preds.append(pred_relevance)
    all_trues.append(true_relevance)

    return all_trues, all_preds


if __name__ == '__main__':
    # queries_id_true_file = sys.argv[1]
    # predicted_file = sys.argv[2]
    # baselines_file = sys.argv[3]

    queries_id_true_file = "D:\\Colecoes\\BD\\web10k-norm\\Fold1\\Norm.test.txt"
    predicted_file = "D:\\Colecoes\\experimento_loss_risk\\resultados2\\zRiskListnetLoss.predict.txt"
    baselines_file = "D:\\Colecoes\\BD\\web10k-norm\\Fold1\\baseline.Norm.test.txt"

    queries, true_relevance_y = read_queries_id_true_relevance(queries_id_true_file)
    predictions_r = read_predicted(predicted_file)

    assert len(queries) == len(true_relevance_y)
    assert len(queries) == len(predictions_r)

    all_trues, all_preds = group_queries(queries, true_relevance_y, predictions_r)

    # ndcg10 = mNdcg(all_trues, all_preds, k=10, no_relevant=True, gains='exponential', use_numpy=True)
    # ndcg5 = mNdcg(all_trues, all_preds, k=5, no_relevant=True, gains='exponential', use_numpy=True)
    # ndcg100 = mNdcg(all_trues, all_preds, k=5, no_relevant=True, gains='exponential', use_numpy=True)

    baselines_by_query = load_baseline_file(baselines_file)

    for k in [5, 10, 100]:
        # ndcgatk = mNdcg(all_trues, -1*np.array(all_preds), k=k)
        ndcgatk = mNdcg(all_trues, np.array(all_preds), k=k)

        num_systems = len(baselines_by_query[0][0])
        # baselines_by_query = np.array(baselines_by_query)
        mat = [ndcgatk]
        for i in range(num_systems):
            mat.append(mNdcg(all_trues, np.array(baselines_by_query)[:, :, i], k=k, gains="exponential", no_relevant=False))
            print(np.mean(mNdcg(all_trues, np.array(baselines_by_query)[:, :, i], k=k, gains="exponential", no_relevant=False)))
        mat = np.array(mat)
        georisks = getGeoRiskDefault(mat, alpha=5)
        print(f"ndcg@{k}: {np.mean(ndcgatk)} -- georisk5: {georisks[0]}")
