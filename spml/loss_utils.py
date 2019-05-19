import torch
import torch.nn.functional as F


def kld_loss(mean, logvar):
    return - 0.5 * (1 + logvar - mean.pow(2) - logvar.exp())


def bce(var, target):
    target = target * torch.ones_like(var, dtype=torch.float)
    return F.binary_cross_entropy(input=var, target=target)
