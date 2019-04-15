#!/usr/bin/env python
"""Run VAE-GAN model to encode-decode frame by frame"""
import argparse

import numpy as np
import torch
import torch.utils.data

import sp_vae_gan.model
import sp_vae_gan.dataloader
from sp_vae_gan import image_util


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


def _encode(buffer_):
    batch = 2 * np.asarray(buffer_, dtype='float') / 255 - 1.0
    return torch.from_numpy(batch)


def _get_loader(input_file, scale, batch_size=32, debug=False):
    dataset = sp_vae_gan.dataloader.VideoDataset(
        input_file, frame_rate=30, frame_dim=scale, debug=debug)
    buffer_ = []
    for frame in dataset:
        buffer_.append(frame)
        if len(buffer_) >= batch_size:
            yield _encode(buffer_)
            buffer_ = []
    if buffer_:
        yield _encode(buffer_)


def _slice(data):
    buffer_ = 255 * (data / 2.0 + 0.5)
    buffer_ = buffer_.astype('uint8').transpose(0, 2, 3, 1)
    for frame in buffer_:
        yield frame.tostring()


def _main():
    args = _parse_args()
    model = _load_model(args.model)
    model = model.float()
    with torch.no_grad():
        saver = image_util.save_video(args.output, (242, 64), debug=args.debug)
        generator = _get_loader(args.video, (121, 65), debug=args.debug)
        for image in generator:
            image = image.float()
            z_mean, _ = model.vae.encoder(image)
            recon = model.vae.decoder(z_mean)

            output = np.concatenate(
                (image.data.numpy(), recon.data.numpy()), axis=3)
            output = output[:, :, :64, :]
            for frame in _slice(output):
                saver.stdin.write(frame)
            saver.stdin.flush()
        saver.stdin.close()


if __name__ == '__main__':
    _main()
