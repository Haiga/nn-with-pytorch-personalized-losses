import torch
import torch.nn.functional as F

from losses.lambdaL import lambdaMask
from losses.riskLosses.riskFunctions import geoRisk, zRisk


def geoRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=0):
    p_y_baselines = F.softmax(y_baselines, dim=1)
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    mat = [p_y_true * torch.log(p_y_predicted)]
    for i in range(p_y_baselines.shape[2]):
        mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
    mat = torch.stack(mat)
    mat = torch.sum(mat, dim=2)
    mat = -1 * mat.t()

    if normalization > 0:
        if normalization == 1:
            mat = mat - torch.min(mat)
            mat = mat / torch.max(mat)
        if normalization == 2:
            mat = mat - torch.mean(mat)
            mat = mat / torch.std(mat)

        # F.softmax(mat)
        # F.sigmoid(mat)

    return geoRisk(mat, alpha, requires_grad=True)

def geoRiskLambdaLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, weighing_scheme="ndcgLoss2PP_scheme"):
    p_y_baselines = torch.squeeze(y_baselines)
    p_y_true = torch.squeeze(y_true)
    p_y_predicted = torch.squeeze(y_predicted)

    p_y_predicted_mask = torch.sum(
        torch.sum(lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True), dim=1),
        dim=1)
    mat = [p_y_predicted_mask]
    for i in range(p_y_baselines.shape[2]):
        p_y_baselines_mask_i = lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme,
                                          return_losses=True)
        p_y_baselines_mask_i = torch.sum(torch.sum(p_y_baselines_mask_i, dim=1), dim=1)
        mat.append(p_y_baselines_mask_i)

    mat = torch.stack(mat)
    # mat = torch.sum(mat, dim=2)
    mat = mat.t()

    if normalization:
        if normalization == 1:
            mat = mat - torch.min(mat)
            mat = mat / torch.max(mat)
        if normalization == 2:
            mat = mat - torch.mean(mat)
            mat = mat / torch.std(mat)

        # F.softmax(mat)
        # F.sigmoid(mat)

    return geoRisk(mat, alpha, requires_grad=True)

def zRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=0):
    p_y_baselines = F.softmax(y_baselines, dim=1)
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    mat = [p_y_true * torch.log(p_y_predicted)]
    for i in range(p_y_baselines.shape[2]):
        mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
    mat = torch.stack(mat)
    mat = torch.sum(mat, dim=2)
    mat = -1 * mat.t()

    if normalization:
        if normalization == 1:
            mat = mat - torch.min(mat)
            mat = mat / torch.max(mat)
        if normalization == 2:
            mat = mat - torch.mean(mat)
            mat = mat / torch.std(mat)

        # F.softmax(mat)
        # F.sigmoid(mat)

    return zRisk(mat, alpha, requires_grad=True)


def zRiskLambdaLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, weighing_scheme="ndcgLoss2PP_scheme"):
    p_y_baselines = torch.squeeze(y_baselines)
    p_y_true = torch.squeeze(y_true)
    p_y_predicted = torch.squeeze(y_predicted)

    p_y_predicted_mask = torch.sum(
        torch.sum(lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True), dim=1),
        dim=1)
    mat = [p_y_predicted_mask]
    for i in range(p_y_baselines.shape[2]):
        p_y_baselines_mask_i = lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme,
                                          return_losses=True)
        p_y_baselines_mask_i = torch.sum(torch.sum(p_y_baselines_mask_i, dim=1), dim=1)
        mat.append(p_y_baselines_mask_i)

    mat = torch.stack(mat)
    # mat = torch.sum(mat, dim=2)
    mat = mat.t()

    if normalization:
        if normalization == 1:
            mat = mat - torch.min(mat)
            mat = mat / torch.max(mat)
        if normalization == 2:
            mat = mat - torch.mean(mat)
            mat = mat / torch.std(mat)

        # F.softmax(mat)
        # F.sigmoid(mat)

    return zRisk(mat, alpha, requires_grad=True)


def tRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=0):
    # p_y_baselines = F.softmax(y_baselines, dim=1)
    p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    p_queries_y_true = -1 * p_y_true * torch.log(p_y_predicted)
    p_queries_y_baselines = -1 * p_y_baselines * torch.log(p_y_predicted)

    p_queries_y_true = torch.sum(p_queries_y_true, dim=1)
    p_queries_y_baselines = torch.sum(p_queries_y_baselines, dim=1)

    if normalization > 0:
        mat = torch.stack([p_queries_y_true, p_queries_y_baselines])
        if normalization == 1:
            mat = mat - torch.min(mat)
            mat = mat / torch.max(mat)
        if normalization == 2:
            mat = mat - torch.mean(mat)
            mat = mat / torch.std(mat)

        p_queries_y_true = mat[0]
        p_queries_y_baselines = mat[1]

    mask = p_queries_y_true < p_queries_y_baselines
    alpha_in = torch.tensor([alpha], requires_grad=True, dtype=torch.float)

    delta = p_queries_y_true - p_queries_y_baselines
    delta = delta * mask * alpha_in + delta

    urisk = torch.mean(delta)
    se_urisk = torch.std(delta)
    return urisk / se_urisk


def tRiskLambdaLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, weighing_scheme="ndcgLoss2PP_scheme"):
    p_y_baselines = torch.squeeze(y_baselines, dim=1)
    p_y_true = torch.squeeze(y_true, dim=1)
    p_y_predicted = torch.squeeze(y_predicted, dim=1)

    p_queries_y_true = lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True)
    p_queries_y_baselines = lambdaMask(p_y_predicted, p_y_baselines, weighing_scheme=weighing_scheme,
                                       return_losses=True)

    p_queries_y_true = torch.sum(torch.sum(p_queries_y_true, dim=1), dim=1)
    p_queries_y_baselines = torch.sum(torch.sum(p_queries_y_baselines, dim=1), dim=1)

    if normalization > 0:
        mat = torch.stack([p_queries_y_true, p_queries_y_baselines])
        if normalization == 1:
            mat = mat - torch.min(mat)
            mat = mat / torch.max(mat)
        if normalization == 2:
            mat = mat - torch.mean(mat)
            mat = mat / torch.std(mat)

        p_queries_y_true = mat[0]
        p_queries_y_baselines = mat[1]

    mask = p_queries_y_true < p_queries_y_baselines
    alpha_in = torch.tensor([alpha], requires_grad=True, dtype=torch.float)

    delta = p_queries_y_true - p_queries_y_baselines
    delta = delta * mask * alpha_in + delta

    urisk = torch.mean(delta)
    se_urisk = torch.std(delta)
    return urisk / se_urisk
