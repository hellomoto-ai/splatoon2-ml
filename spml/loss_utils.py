import torch
import torch.nn.functional as F


def kld_loss(mean, var):
    logvar = torch.log(var.clamp(min=1e-12))
    return - 0.5 * (1 + logvar - mean.pow(2) - var)


def bce(var, target):
    target = target * torch.ones_like(var, dtype=torch.float)
    return F.binary_cross_entropy(input=var, target=target)


class MovingStats:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.mean = 0
        self.var = 1

    def __call__(self, x, update):
        x_mean = torch.mean(x, dim=0)
        x_var = torch.var(x, dim=0)

        mean = self.momentum * self.mean + (1 - self.momentum) * x_mean
        var = self.momentum * self.var + (1 - self.momentum) * x_var

        if update:
            self.mean = mean.detach()
            self.var = var.detach()

        return mean, var
