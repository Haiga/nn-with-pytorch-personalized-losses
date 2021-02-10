import torch
import torch.nn.functional as F

from losses.lambdaL import lambdaMask
from losses.riskLosses.riskFunctions import geoRisk, zRisk


##WOrking 5 - PARECE FUNCIONAR BEM SÒ COM O GEORISK - sem dropout
def geoRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=1, strategy=1, negative=-1):
    p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    p_y_true_2 = p_y_true * p_y_true
    # (p_y_true * p_y_predicted - p_y_true * p_y_true) ** 2

    # mat = [(p_y_true * p_y_predicted - p_y_true_2) ** 2]  # similaridade em vez de distância?
    # torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)
    mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)]
    # for i in range(p_y_baselines.shape[2]):
    #     # mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
    #     # mat.append((p_y_true * p_y_baselines[:, :, i] - p_y_true_2) ** 2)
    #     mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_baselines[:, :, i]))
    mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))

    # essa abordagem deixa como melhor efetividade a maior distância, temos que inverter
    mat = torch.stack(mat)
    # mat = torch.sum(mat, dim=2)
    mat = mat.t()
    imat = mat
    # mat = -mat + torch.max(mat)

    # if normalization > 0:
    #     if normalization == 1:
    #         mat = mat - torch.min(mat)
    #         mat = mat / torch.max(mat)
    #     if normalization == 2:
    #         mat = mat - torch.mean(mat)
    #         mat = mat / torch.std(mat)

    if strategy == 1:

        return geoRisk(mat, alpha, requires_grad=True)
    elif strategy == 2:
        return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
    else:
        return negative * geoRisk(mat, alpha, requires_grad=True)


# #WORKINg4-op - distância para a métrica
# def geoRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=1, strategy=1, negative=-1):
#     p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
#     p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
#     p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
#
#     p_y_true_2 = torch.sum(p_y_true * p_y_true, dim=1)
#     # (p_y_true * p_y_predicted - p_y_true * p_y_true) ** 2
#
#     mat = [(torch.sum(p_y_true * p_y_predicted, dim=1) - p_y_true_2) ** 2]  # similaridade em vez de distância?
#     # torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)
#     for i in range(p_y_baselines.shape[2]):
#         # mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
#         mat.append((torch.sum(p_y_true * p_y_baselines[:, :, i], dim=1) - p_y_true_2) ** 2)
#     # mat.append(p_y_true * F.softmax(p_y_true, dim=1))
#     mat.append((torch.sum(p_y_true *p_y_true, dim=1) - p_y_true_2) ** 2)
#
#     # essa abordagem deixa como melhor efetividade a maior distância, temos que inverter
#     mat = torch.stack(mat)
#     # mat = torch.sum(mat, dim=2)
#     mat = mat.t()
#     imat = mat
#     mat = -mat + torch.max(mat)
#
#     # if normalization > 0:
#     #     if normalization == 1:
#     #         mat = mat - torch.min(mat)
#     #         mat = mat / torch.max(mat)
#     #     if normalization == 2:
#     #         mat = mat - torch.mean(mat)
#     #         mat = mat / torch.std(mat)
#
#     if strategy == 1:
# #those lines
#         # return (geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True))**2
#         return geoRisk(mat, alpha, requires_grad=True)
#     elif strategy == 2:
#         return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#     else:
#         return negative * geoRisk(mat, alpha, requires_grad=True)

##WOrking 3 - cosino com o sem baseline, com -1 e 1
# def geoRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=1, strategy=1, negative=-1):
#     p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
#     p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
#     p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
#
#     p_y_true_2 = p_y_true * p_y_true
#     # (p_y_true * p_y_predicted - p_y_true * p_y_true) ** 2
#
#     # mat = [(p_y_true * p_y_predicted - p_y_true_2) ** 2]  # similaridade em vez de distância?
#     # torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)
#     mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)]
#     # for i in range(p_y_baselines.shape[2]):
#     #     # mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
#     #     # mat.append((p_y_true * p_y_baselines[:, :, i] - p_y_true_2) ** 2)
#     #     mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_baselines[:, :, i]))
#     mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))
#
#     # essa abordagem deixa como melhor efetividade a maior distância, temos que inverter
#     mat = torch.stack(mat)
#     # mat = torch.sum(mat, dim=2)
#     mat = mat.t()
#     imat = mat
#     # mat = -mat + torch.max(mat)
#
#     # if normalization > 0:
#     #     if normalization == 1:
#     #         mat = mat - torch.min(mat)
#     #         mat = mat / torch.max(mat)
#     #     if normalization == 2:
#     #         mat = mat - torch.mean(mat)
#     #         mat = mat / torch.std(mat)
#
#     if strategy == 1:
#         # rmat = torch.cat([torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0), mat.t()]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - geoRisk(rmat.clone(), alpha, requires_grad=True)) ** 2
#         # mat = torch.cat([mat.t(), torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0)]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#         # return geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)
#         return -(geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True))
#     elif strategy == 2:
#         return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#     else:
#         return negative * geoRisk(mat, alpha, requires_grad=True)

#########WORKING 2 similarity
# def geoRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=1, strategy=1, negative=-1):
#     p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
#     p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
#     p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
#
#     p_y_true_2 = p_y_true * p_y_true
#     # (p_y_true * p_y_predicted - p_y_true * p_y_true) ** 2
#
#     # mat = [(p_y_true * p_y_predicted - p_y_true_2) ** 2]  # similaridade em vez de distância?
#     # torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)
#     mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)]
#     for i in range(p_y_baselines.shape[2]):
#         # mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
#         # mat.append((p_y_true * p_y_baselines[:, :, i] - p_y_true_2) ** 2)
#         mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_baselines[:, :, i]))
#     # mat.append(p_y_true * F.softmax(p_y_true, dim=1))
#
#     # essa abordagem deixa como melhor efetividade a maior distância, temos que inverter
#     mat = torch.stack(mat)
#     # mat = torch.sum(mat, dim=2)
#     mat = mat.t()
#     imat = mat
#     # mat = -mat + torch.max(mat)
#
#     # if normalization > 0:
#     #     if normalization == 1:
#     #         mat = mat - torch.min(mat)
#     #         mat = mat / torch.max(mat)
#     #     if normalization == 2:
#     #         mat = mat - torch.mean(mat)
#     #         mat = mat / torch.std(mat)
#
#     if strategy == 1:
#         # rmat = torch.cat([torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0), mat.t()]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - geoRisk(rmat.clone(), alpha, requires_grad=True)) ** 2
#         mat = torch.cat([mat.t(), torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0)]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#         # return geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)
#         return geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True)
#     elif strategy == 2:
#         return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#     else:
#         return negative * geoRisk(mat, alpha, requires_grad=True)

#
##WORKINg1 - distância meio bugada
# def geoRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=1, strategy=1, negative=-1):
#     p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
#     p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
#     p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
#
#     p_y_true_2 = p_y_true * p_y_true
#     # (p_y_true * p_y_predicted - p_y_true * p_y_true) ** 2
#
#     mat = [(p_y_true * p_y_predicted - p_y_true_2) ** 2]  # similaridade em vez de distância?
#     # torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)
#     for i in range(p_y_baselines.shape[2]):
#         # mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
#         mat.append((p_y_true * p_y_baselines[:, :, i] - p_y_true_2) ** 2)
#     # mat.append(p_y_true * F.softmax(p_y_true, dim=1))
#
#     # essa abordagem deixa como melhor efetividade a maior distância, temos que inverter
#     mat = torch.stack(mat)
#     mat = torch.sum(mat, dim=2)
#     mat = mat.t()
#     imat = mat
#     mat = -mat + torch.max(mat)
#
#     # if normalization > 0:
#     #     if normalization == 1:
#     #         mat = mat - torch.min(mat)
#     #         mat = mat / torch.max(mat)
#     #     if normalization == 2:
#     #         mat = mat - torch.mean(mat)
#     #         mat = mat / torch.std(mat)
#
#     if strategy == 1:
#         # rmat = torch.cat([torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0), mat.t()]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - geoRisk(rmat.clone(), alpha, requires_grad=True)) ** 2
#         mat = torch.cat([mat.t(), torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0)]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#         # return geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)
#         return geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True)
#     elif strategy == 2:
#         return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#     else:
#         return negative * geoRisk(mat, alpha, requires_grad=True)


# def geoRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=1, strategy=1, negative=-1):
#     # p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
#     # p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
#     # p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
#
#     p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
#     # p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
#     # p_y_true = torch.squeeze(F.tanh(y_true))
#     p_y_true = torch.squeeze(y_true)
#     p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
#
#     # return -torch.mean(p_y_true * p_y_predicted)
#     # mat = [p_y_true * torch.log(p_y_predicted)]
#     mat = [p_y_true * p_y_predicted]
#     for i in range(p_y_baselines.shape[2]):
#         # mat.append(p_y_true * torch.log(p_y_baselines)[:, :, i])
#         mat.append(p_y_true * p_y_baselines[:, :, i])
#     # mat.append(p_y_true * F.softmax(p_y_true, dim=1))
#     mat = torch.stack(mat)
#     mat = torch.sum(mat, dim=2)
#     mat = mat.t()  # Veficar essa linha - tirei o -1 e melhorou (mas está rodando o double Layer) - sem o double layer fica ruim
#     # allinha do normalization também foi alterada
#     # TODO ver o que está causando o nan
#     # b = nn.MSELoss()
#     # a = nn.CrossEntropyLoss()
#     # Fiz algumas alterações no main e acima também
#     # loss_a = a(output_x, x_labels) #VERIFICAR
#     # loss_b = b(output_y, y_labels)
#
#     if normalization > 0:
#         if normalization == 1:
#             mat = mat - torch.min(mat)
#             mat = mat / torch.max(mat)
#         if normalization == 2:
#             mat = mat - torch.mean(mat)
#             mat = mat / torch.std(mat)
#
#         # F.softmax(mat)
#         # F.sigmoid(mat)
#
#     # return geoRisk(mat, alpha, requires_grad=True)
#
#     if strategy == 1:
#         # rmat = torch.cat([torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0), mat.t()]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - geoRisk(rmat.clone(), alpha, requires_grad=True)) ** 2
#         mat = torch.cat([mat.t(), torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0)]).t()
#         # return (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#         # return geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)
#         return geoRisk(mat, alpha, requires_grad=True, i=-1) - geoRisk(mat, alpha, requires_grad=True)
#     elif strategy == 2:
#         return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
#     else:
#         return negative * geoRisk(mat, alpha, requires_grad=True)


def geoRiskLambdaLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, weighing_scheme="ndcgLoss2PP_scheme",
                      strategy=1, negative=1):
    p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

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

    # return geoRisk(mat, alpha, requires_grad=True)

    if strategy == 1:
        mat = torch.cat([mat.t(), torch.unsqueeze(torch.max(mat, dim=1)[0], dim=0)]).t()
        return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
    elif strategy == 2:
        return negative * (geoRisk(mat, alpha, requires_grad=True) - maxGeoRisk(mat, alpha, requires_grad=True)) ** 2
    else:
        return negative * geoRisk(mat, alpha, requires_grad=True)


def zRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, negative=1):
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

    return negative * zRisk(mat, alpha, requires_grad=True)


def zRiskLambdaLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, weighing_scheme="ndcgLoss2PP_scheme",
                    negative=1):
    p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

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

    return negative * zRisk(mat, alpha, requires_grad=True)


def tRiskListnetLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, negative=1):
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
    return negative * urisk / se_urisk


def tRiskLambdaLoss(y_true, y_predicted, y_baselines, alpha, normalization=0, weighing_scheme="ndcgLoss2PP_scheme",
                    negative=1):
    p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

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
    return negative * urisk / se_urisk
