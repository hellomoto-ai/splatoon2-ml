#!/usr/bin/env python3
import argparse

import torch


def _parse_args():
    parser = argparse.ArgumentParser(
        description='From checkpoint file, remove optimizer for publish.'
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Input checkpoint file.')
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output checkpoint file.')
    return parser.parse_args()


def _main():
    args = _parse_args()
    data = torch.load(args.input, map_location='cpu')
    print('Before')
    print(data.keys())
    print(' - Optimizers:', data['optimizers'].keys())
    data['optimizers'] = {}
    torch.save(data, args.output)

    # Check
    data = torch.load(args.output, map_location='cpu')
    print('After')
    print(data.keys())
    print(' - Optimizers:', data['optimizers'].keys())


if __name__ == '__main__':
    _main()
