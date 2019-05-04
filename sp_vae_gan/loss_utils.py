import torch
import torch.nn.functional as F


def format_loss_header():
    return ' '.join(['%10s'] * 6) % (
        'KLD', 'F_RECON',
        'G_RECON', 'D_REAL', 'D_RECON', '[PIXEL]',
    )


def format_loss_dict(loss):
    return ' '.join(['%10.2e'] * 6) % (
        loss['latent'], loss['feats_recon'],
        loss['gen_recon'], loss['disc_orig'], loss['disc_recon'],
        loss['pixel'],
    )


def kld_loss(z):
    z_mean, z_var = z.mean(dim=1), z.var(dim=1)
    z_logvar = torch.log(z_var.clamp_(min=1e-12))
    return - 0.5 * (1 + z_logvar - z_mean.pow(2) - z_var)


def bce(var, target):
    if target == 1:
        target = torch.ones_like(var, dtype=torch.float)
    elif target == 0:
        target = torch.zeros_like(var, dtype=torch.float)
    else:
        raise ValueError('Unexpected target value: %s' % target)
    return F.binary_cross_entropy(input=var, target=target)
