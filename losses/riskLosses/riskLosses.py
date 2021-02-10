import torch
import torch.nn.functional as F

from losses.lambdaL import lambdaMask
from losses.riskLosses.riskFunctions import geoRisk, zRisk


def geoRiskListnetLoss(y_predicted, y_true, y_baselines=None, alpha=5, listnet_transformation=1, return_strategy=1,
                       negative=1, add_ideal_ranking_to_mat=1):
    if not y_baselines is None:
        p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))

    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    if listnet_transformation == 1:
        p_y_true_2 = p_y_true * p_y_true
        mat = [(p_y_true * p_y_predicted - p_y_true_2) ** 2]
    elif listnet_transformation == 2:
        mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)]
    elif listnet_transformation == 3:
        p_y_true_2 = torch.sum(p_y_true * p_y_true, dim=1)
        mat = [(torch.sum(p_y_true * p_y_predicted, dim=1) - p_y_true_2) ** 2]

    if not y_baselines is None:
        for i in range(p_y_baselines.shape[2]):
            if listnet_transformation == 1:
                mat.append((p_y_true * p_y_baselines[:, :, i] - p_y_true_2) ** 2)
            elif listnet_transformation == 2:
                mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_baselines[:, :, i]))
            elif listnet_transformation == 3:
                mat.append((torch.sum(p_y_true * p_y_baselines[:, :, i], dim=1) - p_y_true_2) ** 2)

    # if add_ideal_ranking_to_mat == 1:# do nothing
    if add_ideal_ranking_to_mat == 2:
        if listnet_transformation == 1:
            mat.append((p_y_true * p_y_true - p_y_true_2) ** 2)  # adicionar 0 diretamente?
        elif listnet_transformation == 2:
            mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))  # adicionar 1 diretamente?
        elif listnet_transformation == 3:
            mat.append((torch.sum(p_y_true * p_y_true, dim=1) - p_y_true_2) ** 2)

    mat = torch.stack(mat)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    mat = mat.t()

    # listnet_transformation==1 or 3 deixa como melhor efetividade a maior distância, temos que inverter
    if listnet_transformation == 1 or listnet_transformation == 3:
        mat = -mat + torch.max(mat)

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float)
    if return_strategy == 1:
        return factor * geoRisk(mat, alpha, requires_grad=True)
    elif return_strategy == 2:
        return factor * (geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True))
    elif return_strategy == 3:
        return factor * ((geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True)) ** 2)

    return None


def geoRiskLambdaLoss(y_predicted, y_true, y_baselines=None, alpha=5, listnet_transformation=1, return_strategy=1,
                      negative=1, add_ideal_ranking_to_mat=1, weighing_scheme="ndcgLoss2PP_scheme"):
    if not y_baselines is None:
        p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))

    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    if listnet_transformation == 1:
        p_y_true_y_true = torch.sum(lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                                    dim=1)
        p_y_predicted_y_true = torch.sum(
            lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)
        mat = [(p_y_predicted_y_true - p_y_true_y_true) ** 2]
    elif listnet_transformation == 2:
        p_y_true_y_true = torch.sum(lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                                    dim=1)
        p_y_predicted_y_true = torch.sum(
            lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)

        mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_predicted_y_true)]

    if not y_baselines is None:
        for i in range(p_y_baselines.shape[2]):
            if listnet_transformation == 1:
                p_y_predicted_i = torch.sum(
                    lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                    dim=1)
                mat.append((p_y_predicted_i - p_y_true_y_true) ** 2)
            elif listnet_transformation == 2:
                p_y_predicted_i = torch.sum(
                    lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                    dim=1)
                mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_predicted_i))

    # if add_ideal_ranking_to_mat == 1:# do nothing
    if add_ideal_ranking_to_mat == 2:
        if listnet_transformation == 1:
            mat.append((p_y_true_y_true - p_y_true_y_true) ** 2)  # adicionar 0 diretamente?
        elif listnet_transformation == 2:
            # mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_true_y_true))  # adicionar 1 diretamente?
            mat.append(torch.ones(p_y_true_y_true.shape[0], dtype=torch.float))  # adicionar 1 diretamente?

    mat = torch.stack(mat)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    mat = mat.t()

    # listnet_transformation==1 deixa como melhor efetividade a maior distância, temos que inverter
    if listnet_transformation == 1:
        mat = -mat + torch.max(mat)

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float)
    if return_strategy == 1:
        return factor * geoRisk(mat, alpha, requires_grad=True)
    elif return_strategy == 2:
        return factor * (geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True))
    elif return_strategy == 3:
        return factor * ((geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True)) ** 2)

    return None


def zRiskListnetLoss(y_predicted, y_true, y_baselines=None, alpha=5, listnet_transformation=1, return_strategy=1,
                     negative=1, add_ideal_ranking_to_mat=1):
    if not y_baselines is None:
        p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))

    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    if listnet_transformation == 1:
        p_y_true_2 = p_y_true * p_y_true
        mat = [(p_y_true * p_y_predicted - p_y_true_2) ** 2]
    elif listnet_transformation == 2:
        mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)]
    elif listnet_transformation == 3:
        p_y_true_2 = torch.sum(p_y_true * p_y_true, dim=1)
        mat = [(torch.sum(p_y_true * p_y_predicted, dim=1) - p_y_true_2) ** 2]

    if not y_baselines is None:
        for i in range(p_y_baselines.shape[2]):
            if listnet_transformation == 1:
                mat.append((p_y_true * p_y_baselines[:, :, i] - p_y_true_2) ** 2)
            elif listnet_transformation == 2:
                mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_baselines[:, :, i]))
            elif listnet_transformation == 3:
                mat.append((torch.sum(p_y_true * p_y_baselines[:, :, i], dim=1) - p_y_true_2) ** 2)

    # if add_ideal_ranking_to_mat == 1:# do nothing
    if add_ideal_ranking_to_mat == 2:
        if listnet_transformation == 1:
            mat.append((p_y_true * p_y_true - p_y_true_2) ** 2)
        elif listnet_transformation == 2:
            mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))
        elif listnet_transformation == 3:
            mat.append((torch.sum(p_y_true * p_y_true, dim=1) - p_y_true_2) ** 2)

    mat = torch.stack(mat)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    mat = mat.t()

    # listnet_transformation==1 or 3 deixa como melhor efetividade a maior distância, temos que inverter
    if listnet_transformation == 1 or listnet_transformation == 3:
        mat = -mat + torch.max(mat)

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float)
    if return_strategy == 1:
        return factor * zRisk(mat, alpha, requires_grad=True)
    elif return_strategy == 2:
        return factor * zRisk(mat, alpha, requires_grad=True, i=-1) - zRisk(mat, alpha, requires_grad=True)
    elif return_strategy == 3:
        return factor * (zRisk(mat, alpha, requires_grad=True, i=-1) - zRisk(mat, alpha, requires_grad=True)) ** 2

    return None


def zRiskLambdaLoss(y_predicted, y_true, y_baselines=None, alpha=5, listnet_transformation=1, return_strategy=1,
                    negative=1, add_ideal_ranking_to_mat=1, weighing_scheme="ndcgLoss2PP_scheme"):
    if not y_baselines is None:
        p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))

    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    if listnet_transformation == 1:
        p_y_true_y_true = torch.sum(lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                                    dim=1)
        p_y_predicted_y_true = torch.sum(
            lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)
        mat = [(p_y_predicted_y_true - p_y_true_y_true) ** 2]
    elif listnet_transformation == 2:
        p_y_true_y_true = torch.sum(lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                                    dim=1)
        p_y_predicted_y_true = torch.sum(
            lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)

        mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_predicted_y_true)]

    if not y_baselines is None:
        for i in range(p_y_baselines.shape[2]):
            if listnet_transformation == 1:
                p_y_predicted_i = torch.sum(
                    lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                    dim=1)
                mat.append((p_y_predicted_i - p_y_true_y_true) ** 2)
            elif listnet_transformation == 2:
                p_y_predicted_i = torch.sum(
                    lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                    dim=1)
                mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_predicted_i))

    # if add_ideal_ranking_to_mat == 1:# do nothing
    if add_ideal_ranking_to_mat == 2:
        if listnet_transformation == 1:
            mat.append((p_y_true_y_true - p_y_true_y_true) ** 2)  # adicionar 0 diretamente?
        elif listnet_transformation == 2:
            mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_true_y_true))  # adicionar 1 diretamente?

    mat = torch.stack(mat)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    mat = mat.t()

    # listnet_transformation==1 deixa como melhor efetividade a maior distância, temos que inverter
    if listnet_transformation == 1:
        mat = -mat + torch.max(mat)

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float)
    if return_strategy == 1:
        return factor * zRisk(mat, alpha, requires_grad=True)
    elif return_strategy == 2:
        return factor * (zRisk(mat, alpha, requires_grad=True, i=-1) - zRisk(mat, alpha, requires_grad=True))
    elif return_strategy == 3:
        return factor * ((zRisk(mat, alpha, requires_grad=True, i=-1) - zRisk(mat, alpha, requires_grad=True)) ** 2)

    return None


def tRiskListnetLoss(y_predicted, y_true, y_baselines, alpha=5, listnet_transformation=1, negative=1):
    # p_y_baselines = F.softmax(y_baselines, dim=1)
    p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    p_queries_y_true = p_y_true * p_y_true
    p_queries_y_predicted = p_y_true * p_y_predicted
    p_queries_y_baselines = p_y_true * p_y_baselines

    mat = []
    if listnet_transformation == 1:
        mat.append((p_queries_y_predicted - p_queries_y_true) ** 2)
        mat.append((p_queries_y_baselines - p_queries_y_true) ** 2)
    elif listnet_transformation == 2:
        mat.append(torch.nn.CosineSimilarity(dim=1)(p_queries_y_true, p_queries_y_predicted))
        mat.append(torch.nn.CosineSimilarity(dim=1)(p_queries_y_true, p_queries_y_baselines))
    elif listnet_transformation == 3:
        p_y_true_2 = torch.sum(p_queries_y_true, dim=1)
        mat.append((torch.sum(p_queries_y_predicted, dim=1) - p_y_true_2) ** 2)
        mat.append((torch.sum(p_queries_y_baselines, dim=1) - p_y_true_2) ** 2)

    mat = torch.stack(mat)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    # mat = mat.t()

    if listnet_transformation == 1:
        mat = -mat + torch.max(mat)

    p_queries_y_true = mat[0]
    p_queries_y_baselines = mat[1]

    mask = p_queries_y_true < p_queries_y_baselines
    alpha_in = torch.tensor([alpha], requires_grad=True, dtype=torch.float)

    delta = p_queries_y_true - p_queries_y_baselines
    delta = delta * mask * alpha_in + delta

    urisk = torch.mean(delta)
    se_urisk = torch.std(delta)

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float)

    return factor * urisk / se_urisk


def tRiskLambdaLoss(y_predicted, y_true, y_baselines, alpha=5, listnet_transformation=1, negative=1,
                    weighing_scheme="ndcgLoss2PP_scheme"):
    # p_y_baselines = F.softmax(y_baselines, dim=1)
    p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    p_queries_y_true = torch.sum(
        lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
        dim=1)
    p_queries_y_predicted = torch.sum(
        lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
        dim=1)
    p_queries_y_baselines = torch.sum(
        lambdaMask(p_y_baselines, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
        dim=1)

    mat = []
    if listnet_transformation == 1:
        mat.append((p_queries_y_predicted - p_queries_y_true) ** 2)
        mat.append((p_queries_y_baselines - p_queries_y_true) ** 2)
    elif listnet_transformation == 2:
        mat.append(torch.nn.CosineSimilarity(dim=1)(p_queries_y_true, p_queries_y_predicted))
        mat.append(torch.nn.CosineSimilarity(dim=1)(p_queries_y_true, p_queries_y_baselines))
    elif listnet_transformation == 3:
        p_y_true_2 = torch.sum(p_queries_y_true, dim=1)
        mat.append((torch.sum(p_queries_y_predicted, dim=1) - p_y_true_2) ** 2)
        mat.append((torch.sum(p_queries_y_baselines, dim=1) - p_y_true_2) ** 2)

    mat = torch.stack(mat)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    # mat = mat.t()

    if listnet_transformation == 1:
        mat = -mat + torch.max(mat)

    p_queries_y_true = mat[0]
    p_queries_y_baselines = mat[1]

    mask = p_queries_y_true < p_queries_y_baselines
    alpha_in = torch.tensor([alpha], requires_grad=True, dtype=torch.float)

    delta = p_queries_y_true - p_queries_y_baselines
    delta = delta * mask * alpha_in + delta

    urisk = torch.mean(delta)
    se_urisk = torch.std(delta)

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float)

    return factor * urisk / se_urisk
