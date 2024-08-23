import torch


def bce_loss(y_pred, y_true):
    return (
        torch.where(y_pred < 0, 0, y_pred)
        - y_pred * y_true
        + torch.log(1 + torch.exp(-torch.abs(y_pred)))
    )


def dice_loss(y_pred, y_true):
    SMOOTH = 1e-8
    y_pred = torch.sigmoid(y_pred)

    intersection = (y_pred * y_true).sum()
    all_square = y_pred.sum() + y_true.sum()

    return 1 - 2 * (intersection) / (all_square + SMOOTH)


def focal_loss(y_pred, y_true, eps=1e-8, gamma=2):
    probs = torch.sigmoid(y_pred)
    return -torch.mean(
        (1 - probs) ** gamma * y_true * torch.log(probs)
        + (1 - y_true) * torch.log(1 - probs)
    )


def custom_loss(y_pred, y_true, alpha=1e-2):
    probs = torch.sigmoid(y_pred)
    return (
        -torch.mean(y_true * torch.log(probs) + (1 - y_true) * torch.log(1 - probs))
        + alpha * torch.pow(y_pred - y_true, 2).mean()
    )
