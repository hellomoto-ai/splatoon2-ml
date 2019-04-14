import torch
import torch.nn.functional as F


def format_loss_header():
    return ' '.join(['%10s'] * 6) % (
        'KLD', 'F_RECON', 'G_RECON', 'D_REAL', 'D_RECON', '[PIXEL]')


def format_loss_dict(loss):
    return ' '.join(['%10.2e'] * 6) % (
        loss['latent'], loss['feats_recon'], loss['gen_recon'],
        loss['disc_orig'], loss['disc_recon'], loss['pixel'])


class Loss:
    def __init__(
            self, pixel, latent,
            feats_recon, disc_orig, disc_recon, gen_recon,
    ):
        self.pixel = pixel
        self.latent = latent
        self.feats_recon = feats_recon
        self.disc_orig = disc_orig
        self.disc_recon = disc_recon
        self.gen_recon = gen_recon

    def to_dict(self):
        return {
            'pixel': self.pixel.item(),
            'latent': self.latent.item(),
            'feats_recon': self.feats_recon.item(),
            'disc_orig': self.disc_orig.item(),
            'disc_recon': self.disc_recon.item(),
            'gen_recon': self.gen_recon.item(),
        }


def kld_loss(mean, logvar):
    return - 0.5 * (1 + logvar - mean.pow(2) - logvar.exp())


def _bce(var, target):
    if target == 1:
        target = torch.ones_like(var, dtype=torch.float)
    elif target == 0:
        target = torch.zeros_like(var, dtype=torch.float)
    else:
        raise ValueError('Unexpected target value: %s' % target)
    return F.binary_cross_entropy(input=var, target=target)


def loss_func(output):
    """Compute various loss

    Parameters
    ----------
    output : ModelOutput

    Returns
    -------
    LossRecord
    """
    pixel = F.mse_loss(output.orig, output.recon)

    f_recon = F.mse_loss(input=output.feats_recon, target=output.feats_orig)

    latent = torch.mean(kld_loss(*output.latent))

    disc_orig = _bce(output.preds_orig, 1)
    disc_recon = _bce(output.preds_recon, 0)
    gen_recon = _bce(output.preds_recon, 1)

    return Loss(
        pixel=pixel,
        latent=latent,
        feats_recon=f_recon,
        disc_orig=disc_orig,
        disc_recon=disc_recon,
        gen_recon=gen_recon,
    )
