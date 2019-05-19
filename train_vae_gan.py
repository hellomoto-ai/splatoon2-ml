#!/usr/bin/env python
"""Train VAE-GAN."""
import os
import csv
import sys
import logging
import argparse

import torch
import numpy as np

import spml.dataloader
import spml.models.vae_gan
import spml.trainer


_LG = logging.getLogger(__name__)
_THIS_DIR = os.path.dirname(__file__)


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--train-flist', required=True,
        help='Path to text file with list of trainig image paths.'
    )
    parser.add_argument(
        '--test-flist', required=True,
        help='Path to text file with list of test image paths.'
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='Directory where train/test images are found.',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=os.path.join(_THIS_DIR, 'results', 'output'),
    )
    parser.add_argument(
        '--epoch', type=int, default=10,
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size',
    )
    parser.add_argument(
        '--checkpoint',
        help='Checkpoint file for resume training',
    )
    parser.add_argument(
        '--debug', action='store_true',
    )
    parser.add_argument(
        '--no-cuda', action='store_true',
    )
    return parser.parse_args()


def _get_samples(num_features, device, batch_size=32, seed=0):
    rng = np.random.RandomState(seed)
    samples = rng.randn(batch_size, num_features)
    return torch.tensor(samples, device=device).float().to(device)


def _load_keyframe_flist(flist):
    ret = []
    with open(flist, 'r') as fileobj:
        for line in fileobj:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            path, _ = line.split('\t')
            ret.append(path)
    return ret


def _get_trainer(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Aspect ratio of Switch is 16:9, so we use 16 x 9 x C as size of encoded
    # features.
    # Each decoding block performs downsampling by x2 then -1. So for level-3
    # convolution, input size is (f(f(f(16))), f(f(f(9)))) = (121, 65)
    # where f(x) = 2x - 1
    scale = (121, 65)
    n_latent = 1024
    train_loader = spml.dataloader.get_dataloader(
        _load_keyframe_flist(args.train_flist), args.data_dir,
        args.batch_size, scale)
    # To output the same image over epochs to observe evolution of model,
    # disable shuffle and fix batch_size to 16 (unless smaller value is
    # provided via command line when batch does not fit on GPU memory)
    test_loader = spml.dataloader.get_dataloader(
        _load_keyframe_flist(args.test_flist), args.data_dir,
        min(16, args.batch_size), scale, shuffle=False,
    )
    model = spml.models.vae_gan.get_model(n_latent)
    opt = torch.optim.Adam
    optimizers = {
        'encoder': opt(model.vae.encoder.parameters(), lr=args.lr),
        'decoder': opt(model.vae.decoder.parameters(), lr=args.lr),
        'discriminator': opt(model.discriminator.parameters(), lr=args.lr),
    }
    samples = _get_samples(n_latent, device)
    trainer = spml.trainer.Trainer(
        model, optimizers, train_loader, test_loader, device, args.output_dir,
        samples=samples,
    )
    if args.checkpoint:
        trainer.load(args.checkpoint)
    return trainer


def _run_main(args):
    trainer = _get_trainer(args)
    _LG.info('\n%s', trainer)
    _LG.info('Batch size: %s', args.batch_size)
    trainer.test()
    trainer.generate()
    for _ in range(args.epoch):
        trainer.train_one_epoch()


def _main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    _init_logger(args.debug, os.path.join(args.output_dir, 'train.log'))

    # Wrap main process so that stack trace shows up in log file.
    try:
        _run_main(args)
    except Exception:  # pylint: disable=broad-except
        _LG.exception('Unexpected error occured.')
        sys.exit(1)


def _init_logger(debug, logfile=None):
    format_ = (
        '%(asctime)s: %(levelname)5s: %(message)s' if not debug else
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(lineno)d %(message)s'
    )

    logging.basicConfig(
        format=format_,
        level=logging.DEBUG if debug else logging.INFO,
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)

    if logfile:
        handler = logging.FileHandler(logfile)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(format_)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)


if __name__ == '__main__':
    _main()
