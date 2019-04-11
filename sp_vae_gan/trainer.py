import os
import time
import logging

import numpy as np
import torch

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


class Trainer:
    def __init__(
            self, model, optimizers,
            train_loader, test_loader,
            device, output_dir,
    ):
        self.model = model.float().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizers = optimizers
        self.device = device
        self.output_dir = output_dir
        fields = [
            'PHASE', 'TIME', 'STEP', 'EPOCH',
            'KLD', 'F_RECON', 'F_FAKE',
            'G_RECON', 'G_FAKE', 'D_REAL', 'D_RECON', 'D_FAKE', 'PIXEL',
        ]
        logfile = open(os.path.join(output_dir, 'result.csv'), 'w')
        self.writer = misc_utils.CSVWriter(fields, logfile)

        self.step = 0
        self.epoch = 0

    def _write(self, phase, loss):
        self.writer.write(
            PHASE=phase, STEP=self.step, EPOCH=self.epoch, TIME=time.time(),
            KLD=loss['latent'],
            F_RECON=loss['feats_recon'], F_FAKE=loss['feats_fake'],
            G_RECON=loss['gen_recon'], G_FAKE=loss['gen_fake'],
            D_REAL=loss['disc_orig'], D_RECON=loss['disc_recon'],
            D_FAKE=loss['disc_fake'], PIXEL=loss['pixel'],
        )

    def _forward(self, orig):
        output = self.model(orig.float().to(self.device))
        loss = loss_utils.loss_func(output)
        return output, loss

    def _update(self, loss):
        # TODO: Try loss-based balancing:
        # http://torch.ch/blog/2015/11/13/gan.html

        # TODO: Try removing fake sampling and
        # run update based on encoded sampling multiple times

        # Feature matching
        self.model.zero_grad()
        (loss.feats_recon + loss.feats_fake).backward(retain_graph=True)
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()

        # Latent update
        self.model.zero_grad()
        loss.latent.backward(retain_graph=True)
        self.optimizers['encoder'].step()

        # Discrimator loss - real image
        self.model.zero_grad()
        loss.disc_orig.backward(retain_graph=True)
        self.optimizers['discriminator'].step()

        # Discriminator loss - sampled image
        self.model.zero_grad()
        loss.disc_recon.backward(retain_graph=True)
        self.optimizers['discriminator'].step()

        # Discriminator loss - fake image
        self.model.zero_grad()
        loss.disc_fake.backward(retain_graph=True)
        self.optimizers['discriminator'].step()

        # Generator loss - sampled image
        self.model.zero_grad()
        loss.gen_recon.backward(retain_graph=True)
        self.optimizers['decoder'].step()

        # Generator loss - fake image
        self.model.zero_grad()
        loss.gen_fake.backward(retain_graph=True)
        self.optimizers['decoder'].step()

        # To free internal gradient buffer.
        # reconstruction generator loss is connected to
        # all the trainable variables.
        loss.gen_recon.backward()

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

    def train(self):
        self.model.train()
        _LG.info('         %s', loss_utils.format_loss_header())
        for i, batch in enumerate(self.train_loader):
            _, loss = self._forward(batch['image'])
            self._update(loss)
            self.step += 1

            loss = loss.to_dict()
            self._write('train', loss)
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
        accum = misc_utils.MeanTracker()
        for i, batch in enumerate(self.test_loader):
            orig, path = batch['image'], batch['path']
            output, loss = self._forward(orig)
            accum.update(loss.to_dict())

            if i % 10 == 0:
                _save_images(
                    (orig[0], output.recon[0]), path[0],
                    self.step, self.output_dir)
        self._write('test', accum)
        _LG.info('         %s', loss_utils.format_loss_dict(accum))

    def __repr__(self):
        opt = '\n'.join([
            '%s: %s' % (key, val) for key, val in self.optimizers.items()
        ])
        return 'Epoch: %d\nStep: %d\nModel: %s\nOptimizers: %s\n' % (
            self.epoch, self.step, self.model, opt
        )
