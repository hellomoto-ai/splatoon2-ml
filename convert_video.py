#!/usr/bin/env python
"""Run VAE-GAN model to encode-decode frame by frame"""
import logging
import argparse
import tempfile
import subprocess

import numpy as np
import torch
import torch.utils.data

import sp_vae_gan.model
import sp_vae_gan.dataloader
from sp_vae_gan import image_util

_LG = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('--video', '-v', required=True)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _load_model(path):
    data = torch.load(path, map_location='cpu')
    model = sp_vae_gan.model.get_model()
    model.load_state_dict(data['model'])
    return model


def _combine(image, recon):
    output = np.concatenate((image, recon), axis=3)
    output = output[:, :, :64, :]
    output = 255 * (output / 2.0 + 0.5)
    return output.astype('uint8').transpose(0, 2, 3, 1)


def _copy_audio(src_audio, video, output):
    command = [
        'ffmpeg',
        '-hide_banner', '-y', '-loglevel', 'panic',
        '-i', video, '-i', src_audio,
        '-codec', 'copy', '-map', '0:v:0', '-map', '1:a:0',
        output
    ]
    subprocess.run(command, check=True)


def _main():
    args = _parse_args()
    _init_logger(args.debug)
    _LG.info('Loading model from %s', args.model)
    model = _load_model(args.model).float()
    _LG.info('Openinig video %s', args.video)
    frame_generator = sp_vae_gan.dataloader.VideoSlicer(
        args.video, frame_rate=30, frame_dim=(121, 65), debug=args.debug)
    with tempfile.NamedTemporaryFile('wb', suffix='.mp4') as tmp:
        saver = image_util.VideoSaver(tmp.name, (242, 64), debug=args.debug)
        with saver, torch.no_grad():
            for image in frame_generator:
                mu, _ = model.vae.encoder(image.float())
                recon = model.vae.decoder(mu)
                for frame in _combine(image.numpy(), recon.numpy()):
                    saver.write(frame)
                saver.flush()
        _LG.info('Saving video %s', args.output)
        _copy_audio(args.video, tmp.name, args.output)


def _init_logger(debug):
    format_ = (
        '%(asctime)s: %(levelname)5s: %(message)s' if not debug else
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(lineno)d %(message)s'
    )

    logging.basicConfig(
        format=format_,
        level=logging.DEBUG if debug else logging.INFO,
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)


if __name__ == '__main__':
    _main()
