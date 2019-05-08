import torch
import torch.nn.functional as F


def format_loss_header():
    return ' '.join(['%10s'] * 9) % (
        'KLD', 'BETA', 'F_RECON',
        'G_RECON', 'G_FAKE', 'D_REAL', 'D_RECON', 'D_FAKE', '[PIXEL]',
    )


def format_loss_dict(loss):
    return ' '.join(['%10.2e'] * 9) % (
        loss['kld'], loss['beta'], loss['feats_recon'],
        loss['gen_recon'], loss['gen_fake'],
        loss['disc_orig'], loss['disc_recon'], loss['disc_fake'],
        loss['pixel'],
    )


def kld_loss(samples):
    mean = samples.mean(dim=0)
    var = samples.var(dim=0)
    logvar = torch.log(var.clamp_(min=1e-12))
    return - 0.5 * (1 + logvar - mean.pow(2) - var)


def bce(var, target):
    if target == 1:
        target = torch.ones_like(var, dtype=torch.float)
    elif target == 0:
        target = torch.zeros_like(var, dtype=torch.float)
    else:
        raise ValueError('Unexpected target value: %s' % target)
    return F.binary_cross_entropy(input=var, target=target)
