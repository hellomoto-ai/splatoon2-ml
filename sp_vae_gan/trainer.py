import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

from sp_vae_gan import (
    image_util,
    misc_utils,
    loss_utils,
)

_LG = logging.getLogger(__name__)


def _ensure_dir(filepath):
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)


def _save_images(images, src_path, step, output_dir):
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    save_path = os.path.join(
        output_dir, 'images', src_name, 'step_%d.png' % step)
    _ensure_dir(save_path)

    images = [img.detach().to('cpu').numpy() for img in images]
    images = np.concatenate(images, axis=1)
    image_util.save_image(images, save_path)


def _fetch_numpy(variable):
    return variable.cpu().detach().numpy()


def _get_latent_stats(z_mean, z_logvar):
    z_mean = z_mean.detach()
    z_logvar = z_logvar.detach()
    # Distance from origin
    z_mean = torch.norm(z_mean, dim=1).cpu().numpy()
    # Mean std deviation over latent dimensions
    z_std = torch.exp(0.5 * z_logvar)
    z_var = torch.mean(z_std, dim=1).cpu().numpy()
    # Take average, min, max over batch
    return {
        'z_mean': np.mean(z_mean),
        'z_mean_min': np.min(z_mean),
        'z_mean_max': np.max(z_mean),
        'z_var': np.mean(z_var),
        'z_var_min': np.min(z_var),
        'z_var_max': np.max(z_var),
    }


class Trainer:
    def __init__(
            self, model, optimizers,
            train_loader, test_loader,
            device, output_dir,
            beta=1.0,
    ):
        self.model = model.float().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizers = optimizers
        self.device = device
        self.output_dir = output_dir
        fields = [
            'PHASE', 'TIME', 'STEP', 'EPOCH', 'KLD', 'F_RECON',
            'G_RECON', 'D_REAL', 'D_RECON', 'PIXEL',
            'Z_MEAN', 'Z_MEAN_MIN', 'Z_MEAN_MAX',
            'Z_VAR', 'Z_VAR_MIN', 'Z_VAR_MAX',
        ]
        logfile = open(os.path.join(output_dir, 'result.csv'), 'w')
        self.writer = misc_utils.CSVWriter(fields, logfile)

        self.step = 0
        self.epoch = 0

        self.beta = beta

    def _write(self, phase, loss, stats):
        self.writer.write(
            PHASE=phase, STEP=self.step, EPOCH=self.epoch, TIME=time.time(),
            KLD=loss['latent'],
            F_RECON=loss['feats_recon'],
            G_RECON=loss['gen_recon'], D_REAL=loss['disc_orig'],
            D_RECON=loss['disc_recon'], PIXEL=loss['pixel'],
            Z_MEAN=stats['z_mean'], Z_VAR=stats['z_var'],
            Z_MEAN_MIN=stats['z_mean_min'], Z_VAR_MIN=stats['z_var_min'],
            Z_MEAN_MAX=stats['z_mean_max'], Z_VAR_MAX=stats['z_var_max'],
        )

    def save(self):
        filename = 'epoch_%s_step_%s.pt' % (self.epoch, self.step)
        output = os.path.join(self.output_dir, 'checkpoints', filename)

        _LG.info('Saving checkpoint at %s', output)
        _ensure_dir(output)
        torch.save({
            'model': self.model.state_dict(),
            'optimizers': {
                key: opt.state_dict()
                for key, opt in self.optimizers.items()
            },
            'epoch': self.epoch,
            'step': self.step,
        }, output)

    def load(self, checkpoint):
        _LG.info('Loading checkpoint from %s', checkpoint)
        data = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(data['model'])
        for key, opt in data['optimizers'].items():
            self.optimizers[key].load_state_dict(opt)
        self.epoch = data['epoch']
        self.step = data['step']

    def _forward_gan(self, orig, update=False):
        # TODO: Try loss-based balancing:
        # http://torch.ch/blog/2015/11/13/gan.html

        # 1. Update discriminator with original (real) image
        preds_orig, _ = self.model.discriminator(orig)
        disc_loss_orig = loss_utils.bce(preds_orig, 1)
        if update:
            self.model.zero_grad()
            disc_loss_orig.backward()
            self.optimizers['discriminator'].step()

        # 2. Update discriminator with reconstructed (fake) image
        recon, _ = self.model.vae(orig)
        preds_recon, _ = self.model.discriminator(recon.detach())
        disc_loss_recon = loss_utils.bce(preds_recon, 0)
        if update:
            self.model.zero_grad()
            disc_loss_recon.backward()
            self.optimizers['discriminator'].step()

        # 3. Update generator
        preds_recon, _ = self.model.discriminator(recon)
        gen_loss = loss_utils.bce(preds_recon, 1)
        if update:
            self.model.zero_grad()
            gen_loss.backward()
            self.optimizers['decoder'].step()

        return {
            'disc_orig': disc_loss_orig.item(),
            'disc_recon': disc_loss_recon.item(),
            'gen_recon': gen_loss.item(),
        }

    def _forward_vae(self, orig, update=False):
        # Update feature
        recon, _ = self.model.vae(orig)
        _, feats_orig = self.model.discriminator(orig)
        _, feats_recon = self.model.discriminator(recon)
        feats_loss = F.mse_loss(input=feats_recon, target=feats_orig)
        if update:
            self.model.zero_grad()
            feats_loss.backward()
            self.optimizers['encoder'].step()
            self.optimizers['decoder'].step()

        # Update latent
        z_mean, z_logvar = self.model.vae.encoder(orig)
        latent_loss = torch.mean(loss_utils.kld_loss(z_mean, z_logvar))
        if update:
            beta_latent_loss = self.beta * latent_loss
            self.model.zero_grad()
            beta_latent_loss.backward()
            self.optimizers['encoder'].step()

        loss = {
            'latent': latent_loss.item(),
            'feats_recon': feats_loss.item(),
        }
        stats = _get_latent_stats(z_mean, z_logvar)
        return recon, loss, stats

    def _get_pixel_loss(self, orig):
        recon, _ = self.model.vae(orig)
        return F.mse_loss(orig, recon)

    def _forward(self, orig, update=False):
        loss_gan = self._forward_gan(orig, update=update)
        recon, loss_vae, stats = self._forward_vae(orig, update=update)
        with torch.no_grad():
            pixel_loss = self._get_pixel_loss(orig)

        loss = {'pixel': pixel_loss.item()}
        loss.update(loss_vae)
        loss.update(loss_gan)
        return recon, loss, stats

    def train(self):
        self.model.train()
        _LG.info('         %s', loss_utils.format_loss_header())
        for i, batch in enumerate(self.train_loader):
            orig = batch['image'].float().to(self.device)
            _, loss, stats = self._forward(orig, update=True)
            self.step += 1
            self._write('train', loss, stats)
            if i % 30 == 0:
                progress = 100. * i / len(self.train_loader)
                _LG.info(
                    '  %3d %%: %s',
                    progress, loss_utils.format_loss_dict(loss))
        self.epoch += 1

    def test(self):
        with torch.no_grad():
            self._test()

    def _test(self):
        self.model.eval()
        loss_tracker = misc_utils.StatsTracker()
        stats_tracker = misc_utils.StatsTracker()
        for i, batch in enumerate(self.test_loader):
            orig, path = batch['image'].float().to(self.device), batch['path']
            recon, loss, stats = self._forward(orig, update=False)
            loss_tracker.update(loss)
            stats_tracker.update(stats)
            if i % 10 == 0:
                _save_images(
                    (orig[0], recon[0]), path[0],
                    self.step, self.output_dir)

        self._write('test', loss_tracker, stats)
        _LG.info('         %s', loss_utils.format_loss_dict(loss_tracker))

    def __repr__(self):
        opt = '\n'.join([
            '%s: %s' % (key, val) for key, val in self.optimizers.items()
        ])
        return 'Epoch: %d\nStep: %d\nModel: %s\nOptimizers: %s\nBeta: %s\n' % (
            self.epoch, self.step, self.model, opt, self.beta
        )
