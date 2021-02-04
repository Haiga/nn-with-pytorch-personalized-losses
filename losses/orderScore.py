import torch
import torch.nn.functional as F


def orderScoreLoss(y_true, y_predicted):
    """
    Todo testar orderScoreLoss
    """

    y_true = F.softmax(y_true, dim=1)
    y_predicted = F.softmax(y_predicted, dim=1)
    y_predicted = torch.log(y_predicted)

    corrected_order_y_true = torch.argsort(y_true, descending=True, dim=1)
    ordered_scores_y_predicted = []

    for i in range(y_predicted.shape[0]):
        ordered_scores_y_predicted.append(torch.squeeze(y_predicted[i][corrected_order_y_true[i]]))

    ordered_scores_y_predicted = torch.stack(ordered_scores_y_predicted)
    decrescing_gains = torch.arange(-y_predicted.shape[1], y_predicted.shape[1], 2, dtype=torch.float,
                                    requires_grad=True)
    dgc = decrescing_gains * ordered_scores_y_predicted
    return -torch.mean(torch.sum(dgc, dim=1))
