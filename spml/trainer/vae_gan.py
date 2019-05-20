"""Training mechanism for VAE-GAN"""
import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

from spml import (
    image_util,
    loss_utils,
)
from . import (
    misc_utils,
    saved_model_manager,
)

_LG = logging.getLogger(__name__)


def _save_images(images, src_path, step, output_dir):
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    save_path = os.path.join(
        output_dir, 'images', src_name, 'step_%d.png' % step)
    misc_utils.ensure_dir(save_path)

    images = [img.detach().cpu().numpy() for img in images]
    images = np.concatenate(images, axis=1)
    image_util.save_image(images, save_path)


def _log_header():
    fields = ' '.join(['%10s'] * 9) % (
        'KLD', 'BETA', 'F_RECON',
        'G_RECON', 'G_FAKE', 'D_REAL', 'D_RECON', 'D_FAKE', '[PIXEL]',
    )
    _LG.info('%5s %5s: %s', '', 'PHASE', fields)


_LOGGED = {'last': 0}


def _log_loss(loss, phase, progress=None):
    if _LOGGED['last'] % 30 == 0:
        _log_header()
    _LOGGED['last'] += 1

    header = '' if progress is None else '%3d %%' % progress
    fields = ' '.join(['%10.2e'] * 9) % (
        loss['kld'], loss['beta'], loss['feats_recon'],
        loss['gen_recon'], loss['gen_fake'],
        loss['disc_orig'], loss['disc_recon'], loss['disc_fake'],
        loss['pixel'],
    )
    _LG.info('%5s %5s: %s', header, phase, fields)


def _get_latent_stats(z):
    z = z.detach().cpu().numpy()
    return {
        'z_mean': np.mean(z),
        'z_min': np.min(z),
        'z_max': np.max(z),
        'z_var': np.var(z),
    }


class Trainer:
    def __init__(
            self, model, optimizers,
            train_loader, test_loader,
            device, output_dir,
            initial_beta=100.0,
            beta_step=0.1,
            target_kld=0.1,
            samples=None,
    ):
        self.model = model.float().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizers = optimizers
        self.device = device
        self.output_dir = output_dir

        self.beta = initial_beta
        self.beta_step = beta_step
        self.target_kld = target_kld

        self.samples = samples

        self.saved_model_manager = saved_model_manager.SavedModelManager()

        fields = [
            'PHASE', 'TIME', 'STEP', 'EPOCH', 'KLD', 'BETA', 'F_RECON',
            'G_RECON', 'G_FAKE', 'D_REAL', 'D_RECON', 'D_FAKE', 'PIXEL',
            'Z_MEAN', 'Z_MIN', 'Z_MAX', 'Z_VAR',
        ]
        logfile = open(os.path.join(output_dir, 'result.csv'), 'w')
        self.writer = misc_utils.CSVWriter(fields, logfile)

        self.step = 0
        self.epoch = 0

        self.latent_stats = loss_utils.MovingStats()

    def _write(self, phase, loss, stats):
        self.writer.write(
            PHASE=phase, STEP=self.step, EPOCH=self.epoch, TIME=time.time(),
            KLD=loss['kld'], BETA=loss['beta'],
            F_RECON=loss['feats_recon'],
            G_RECON=loss['gen_recon'], G_FAKE=loss['gen_fake'],
            D_REAL=loss['disc_orig'],
            D_RECON=loss['disc_recon'], D_FAKE=loss['disc_fake'],
            PIXEL=loss['pixel'],
            Z_MEAN=stats['z_mean'], Z_VAR=stats['z_var'],
            Z_MIN=stats['z_min'], Z_MAX=stats['z_max'],
        )

    def save(self):
        filename = 'epoch_%s_step_%s.pt' % (self.epoch, self.step)
        output = os.path.join(self.output_dir, 'checkpoints', filename)

        _LG.info('Saving checkpoint at %s', output)
        misc_utils.ensure_dir(output)
        torch.save({
            'model': self.model.state_dict(),
            'optimizers': {
                key: opt.state_dict()
                for key, opt in self.optimizers.items()
            },
            'epoch': self.epoch,
            'step': self.step,
        }, output)
        return output

    def manage_saved(self, path, loss):
        path = self.saved_model_manager.update(path, loss)
        if path:
            os.remove(path)

    def load(self, checkpoint):
        _LG.info('Loading checkpoint from %s', checkpoint)
        data = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(data['model'])
        for key, opt in data['optimizers'].items():
            self.optimizers[key].load_state_dict(opt)
        self.epoch = data['epoch']
        self.step = data['step']

    def _forward_gan(self, orig, update=False):
        # Update discriminator with original image
        preds_orig, _ = self.model.discriminator(orig)
        disc_loss_orig = loss_utils.bce(preds_orig, 1)
        if update:
            self.model.zero_grad()
            disc_loss_orig.backward()
            self.optimizers['discriminator'].step()

        # Update discriminator with reconstructed image
        recon, latent = self.model.vae(orig)
        preds_recon, _ = self.model.discriminator(recon.detach())
        disc_loss_recon = loss_utils.bce(preds_recon, 0)
        if update:
            self.model.zero_grad()
            disc_loss_recon.backward()
            self.optimizers['discriminator'].step()

        # Update generator with reconstructed image
        preds_recon, _ = self.model.discriminator(recon)
        gen_loss_recon = loss_utils.bce(preds_recon, 1)
        if update:
            self.model.zero_grad()
            gen_loss_recon.backward()
            self.optimizers['decoder'].step()

        # Update discriminator with fake image
        sample = torch.randn_like(latent[0], requires_grad=True)
        fake = self.model.vae.decoder(sample)
        preds_fake, _ = self.model.discriminator(fake.detach())
        disc_loss_fake = loss_utils.bce(preds_fake, 0)
        if update:
            self.model.zero_grad()
            disc_loss_fake.backward()
            self.optimizers['discriminator'].step()

        # Update generator with fake image
        preds_fake, _ = self.model.discriminator(fake)
        gen_loss_fake = loss_utils.bce(preds_fake, 1)
        if update:
            self.model.zero_grad()
            gen_loss_fake.backward()
            self.optimizers['decoder'].step()

        return {
            'disc_orig': disc_loss_orig.item(),
            'disc_recon': disc_loss_recon.item(),
            'disc_fake': disc_loss_fake.item(),
            'gen_recon': gen_loss_recon.item(),
            'gen_fake': gen_loss_fake.item(),
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

        # KLD
        z_mean, z_logvar = self.model.vae.encoder(orig)
        z_std = torch.exp(0.5 * z_logvar)
        latent = z_mean + z_std * torch.randn_like(z_std)
        latent_stats = self.latent_stats(latent, update=update)
        kld = torch.mean(loss_utils.kld_loss(*latent_stats))
        if update:
            beta_latent_loss = self.beta * kld
            self.model.zero_grad()
            beta_latent_loss.backward()
            self.optimizers['encoder'].step()

        # Adjust beta
        if update:
            kld_error = kld.item() - self.target_kld
            self.beta += self.beta_step * kld_error
            self.beta = max(1e-3, self.beta)

        loss = {
            'kld': kld.item(),
            'beta': self.beta,
            'feats_recon': feats_loss.item(),
        }
        stats = _get_latent_stats(latent[0])
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

    def train_batch(self, batch):
        self.model.train()
        orig = batch['image'].float().to(self.device)
        _, loss, stats = self._forward(orig, update=True)
        self._write('train', loss, stats)
        return loss

    def test(self):
        with torch.no_grad():
            return self._test()

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
        self._write('test', loss_tracker, stats_tracker)
        _log_loss(loss_tracker, phase='Test')
        return loss_tracker

    def generate(self, samples=None):
        samples = self.samples if samples is None else samples
        with torch.no_grad():
            self._generate(samples)

    def _generate(self, samples):
        self.model.eval()
        recons = self.model.vae.decoder(samples)
        for i, recon in enumerate(recons):
            path = 'sample_%d.png' % i
            _save_images([recon], path, self.step, self.output_dir)

    def train_one_epoch(self, report_every=180, test_interval=1000):
        last_report = 0
        for i, batch in enumerate(self.train_loader):
            loss = self.train_batch(batch)
            self.step += 1
            if time.time() - last_report > report_every:
                progress = 100. * i / len(self.train_loader)
                _log_loss(loss, 'Train', progress)
                last_report = time.time()
            if self.step % test_interval == 0:
                self.generate()
                loss = self.test()
                path = self.save()
                self.manage_saved(path, loss['pixel'])
        self.epoch += 1

    def __repr__(self):
        opt = '\n'.join([
            '%s: %s' % (key, val) for key, val in self.optimizers.items()
        ])
        beta = '\n'.join([
            'Beta: %s' % self.beta,
            'Beta Step: %s' % self.beta_step,
            'Target KLD: %s' % self.target_kld,
        ])
        return 'Epoch: %d\nStep: %d\nModel: %s\nOptimizers: %s\nBeta: %s\n' % (
            self.epoch, self.step, self.model, opt, beta
        )
