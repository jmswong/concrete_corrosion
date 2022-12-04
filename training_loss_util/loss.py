"""Loss function utils"""

import torch


def weighted_loss(pred, y, loss_fn, pos_weight=1):
    '''
    Computes a weighted version of a given binary loss_fn, weighing the
    positive class by pos_weight (sometimes referred to as alpha).

    weighted_loss = pos_weight * loss if y == 1 else loss

    Args:
        pred (tensor): Dim (num_samples, 1) tensor of predictions.
        y (tensor): Dim (num_samples) tensor of true labels.
        loss_fn (lambda): Binary loss function (e.g. BCELoss).
        pos_weight: Weight applied to positive samples.

    Returns:
        avg_loss (tensor): Scalar, average weighted loss.
    '''
    y = y.unsqueeze(-1)
    loss = loss_fn(pred, y)

    # This constructs a weight vector where the element is pos_weight when
    # y == 1 and 1 otherwise.
    weights = (y * (pos_weight - 1) + 1)

    avg_loss = torch.sum(loss * weights) / sum(weights)
    return avg_loss
