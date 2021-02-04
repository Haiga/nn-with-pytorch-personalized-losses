import torch
import torch.nn.functional as F


def ndcgLoss(y_true, y_predicted):
    """
        Todo loss not differentiable
    """
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
        ordered_scores_y_true.append(torch.squeeze(y_true[i][corrected_order[i]]))

    ordered_scores_y_true = torch.stack(ordered_scores_y_true)
    idcg = torch.sum(ordered_scores_y_true / inverted_gains, dim=1)

    return torch.mean(-(dcg / idcg))
