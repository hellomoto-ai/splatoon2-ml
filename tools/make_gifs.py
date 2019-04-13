#!/usr/bin/env python
import os
import imageio


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i', required=True,
        help='Input directory')
    parser.add_argument(
        '--output', '-o', default='out.gif',
        help='Output file path.')
    parser.add_argument(
        '--duration', '-d', default=0.08, type=float,
    )
    return parser.parse_args()


def _get_suffix(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    index = basename.split('_')[-1]
    return int(index)


def _main():
    args = _parse_args()

    files = [
        os.path.join(args.input, filename)
        for filename in os.listdir(args.input)
    ]
    files.sort(key=_get_suffix)
    files = files[:60]
    images = [imageio.imread(f) for f in files]
    duration = [args.duration for _ in range(len(images))]
    duration[0] = 1.0
    duration[-1] = 3.0
    imageio.mimsave(args.output, images, duration=duration)


if __name__ == '__main__':
    _main()
