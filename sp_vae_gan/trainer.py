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
            initial_beta=1.0,
            beta_step=0.1,
            target_kld=0.35,
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

        fields = [
            'PHASE', 'TIME', 'STEP', 'EPOCH', 'KLD', 'BETA', 'F_RECON',
            'G_RECON', 'G_FAKE', 'D_REAL', 'D_RECON', 'D_FAKE', 'PIXEL',
            'Z_MEAN', 'Z_MIN', 'Z_MAX', 'Z_VAR',
        ]
        logfile = open(os.path.join(output_dir, 'result.csv'), 'w')
        self.writer = misc_utils.CSVWriter(fields, logfile)

        self.step = 0
        self.epoch = 0

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
        # Update discriminator with original image
        preds_orig, _ = self.model.discriminator(orig)
        disc_loss_orig = loss_utils.bce(preds_orig, 1)
        if update:
            self.model.zero_grad()
            disc_loss_orig.backward()
            self.optimizers['discriminator'].step()

        # Update discriminator with reconstructed image
        recon, sample = self.model.vae(orig)
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
        sample = torch.randn_like(sample, requires_grad=True)
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
        sample = self.model.vae.encoder(orig)
        kld = torch.mean(loss_utils.kld_loss(sample))
        if update:
            beta_latent_loss = self.beta * kld
            self.model.zero_grad()
            beta_latent_loss.backward()
            self.optimizers['encoder'].step()

        # Adjust beta
        if update:
            # Use hinge loss
            kld_error = max(0, kld.item() - self.target_kld)
            self.beta += self.beta_step * kld_error

        loss = {
            'kld': kld.item(),
            'beta': self.beta,
            'feats_recon': feats_loss.item(),
        }
        stats = _get_latent_stats(sample)
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

    def generate(self, samples):
        with torch.no_grad():
            self._generate(samples)

    def _generate(self, samples):
        self.model.eval()
        recons = self.model.vae.decoder(samples)
        for i, recon in enumerate(recons):
            path = 'sample_%d.png' % i
            _save_images([recon], path, self.step, self.output_dir)

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
