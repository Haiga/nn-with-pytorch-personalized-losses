import torch
import torch.nn.functional as F


def listnetLoss(y_true, y_predicted, apply_sigmoid=False):
    """
        Default ListNet loss
        apply_sigmoid - adicionei como um ajuste
        Todo testar apply_sigmoid
    """
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    if apply_sigmoid:
        r = F.sigmoid(p_y_true * torch.log(p_y_predicted))
        return -torch.sum(r)
    return - torch.sum(p_y_true * torch.log(p_y_predicted))
