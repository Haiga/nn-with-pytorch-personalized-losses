from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, ndcg_score
import numpy as np
from metrics import mNdcg

mem = Memory("./mycache")


@mem.cache
def get_data(t):
    # data = load_svmlight_file("D:\\Colecoes\\2003_td_dataset\\Fold1\\"+t+".txt", query_id=True)
    data = load_svmlight_file("D:\\Colecoes\\web10k\\Fold1\\Norm." + t + ".txt", query_id=True)
    return data[0], data[1], data[2]


X, y, queries_id = get_data("train")
# model = RandomForestRegressor(n_estimators=3, n_jobs=-1, verbose=1)
model = LinearRegression()
# model = DecisionTreeRegressor()
# model = LogisticRegression()
model.fit(X, y)

X, y, queries_id = get_data("test")
p = model.predict(X)

print(mean_squared_error(y, p))

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

for t in ['linear']:
    # for n in [True, False]:
    for n in [True]:
        for k in [1, 5, 10]:
            for u in [True, False]:
                x = mNdcg(all_trues, all_preds, k=k, no_relevant=n, gains=t, use_numpy=u)
                # print(len(x))
                mx = np.mean(x)
                print(f"NDCG@{k}: {mx} --- numpy{u}")

min_cont = min(conts)
max_cont = max(conts)

print(f"Querie with min doc has {min_cont} docs")
print(f"Querie with max doc has {max_cont} docs")

threshold = 30

cont = 0
all_choices = []
conc_choices = []
discarded_queries = 0
for i in range(len(queries_id_unique)):
    if conts[i] >= threshold:
        choices = np.random.choice(np.arange(cont, cont + conts[i]), threshold, replace=False)
        all_choices.append(choices)
        conc_choices = np.concatenate([conc_choices, choices])
    else:
        discarded_queries += 1
    cont = cont + conts[i]
print("-------------------")
print(f"{discarded_queries} queries has been discarded")
conc_choices = np.array(conc_choices, dtype=np.int32)
X = X.toarray()
X = X[conc_choices, 0:X.shape[1]]
y = y[conc_choices]
queries_id = queries_id[conc_choices]

p = model.predict(X)
print(mean_squared_error(y, p))


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


#######
# for t in ['exponential', 'linear']:
for t in ['linear']:
    # for n in [True, False]:
    for n in [True]:
        for k in [1, 5, 10]:
            for u in [True, False]:
                x = mNdcg(all_trues, all_preds, k=k, no_relevant=n, gains=t, use_numpy=u)
                # print(len(x))
                mx = np.mean(x)
                print(f"NDCG@{k}: {mx} --- numpy{u}")

print("-------------------")
k = 5
x = mNdcg(all_trues, all_preds, k=k, no_relevant=False, gains="linear", use_numpy=True)
# print(len(x))
mx = np.mean(x)
print(f"NDCG@{k}: {mx}")
mx = ndcg_score(all_trues, all_preds, k=k)
print(f"scikit NDCG@{k}: {mx}\n-----\n")


k = 10
x = mNdcg(all_trues, all_preds, k=k, no_relevant=False, gains="linear", use_numpy=True)
# print(len(x))
mx = np.mean(x)
print(f"NDCG@{k}: {mx}")
mx = ndcg_score(all_trues, all_preds, k=k)
print(f"scikit NDCG@{k}: {mx}")