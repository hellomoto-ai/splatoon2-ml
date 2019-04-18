
#!/usr/bin/env python
from collections import namedtuple

import pandas
import numpy as np
import matplotlib.pyplot as plt


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i', required=True, nargs=2,
        help='2 CSVs. Base result and new result.'
    )
    return parser.parse_args()


Data = namedtuple('Data', ['train', 'test'])


def _load(path):
    data = pandas.read_csv(path)
    train_data = data[data['PHASE'] == 'train']
    test_data = data[data['PHASE'] == 'test']
    return Data(train_data, test_data)


def _avg(data, n=100):
    return [np.mean(data[i:i+n]) for i in range(0, len(data), n)]


def _plot_envelopped(ax, x, y, label, color, linestyle='-'):
    ax.plot(x, y, label='_nolegend_', color=color, alpha=0.25)
    ax.plot(_avg(x), _avg(y), label=label, color=color, linestyle=linestyle)


def _plot_klds(base_data, new_data):
    fig, ax = plt.subplots()
    _plot_envelopped(
        ax, base_data.train['STEP'], base_data.train['KLD'],
        label='Training w/ random sample', color='b')
    ax.plot(
        base_data.test['STEP'], base_data.test['KLD'],
        label='Test w/ random sample', color='c')
    _plot_envelopped(
        ax, new_data.train['STEP'], new_data.train['KLD'],
        label='Training w/o random sample', color='r')
    ax.plot(
        new_data.test['STEP'], new_data.test['KLD'],
        label='Test w/o random sample', color='m')
    ax.grid()
    ax.legend()
    ax.set_ylim(0, 12)
    ax.set(xlabel='Steps', ylabel='KLD', title='KL Divergence')
    return fig


def _plot_gan_axis(ax, data, colors, prefix, avg=True):
    step = _avg(data['STEP']) if avg else data['STEP']
    real = _avg(data['D_REAL']) if avg else data['D_REAL']
    d_recon = _avg(data['D_RECON']) if avg else data['D_RECON']
    g_recon = _avg(data['G_RECON']) if avg else data['G_RECON']
    ax.plot(step, real, label='%s - D(x)' % prefix, color=colors[0], alpha=0.9)
    ax.plot(step, d_recon, label='%s - D(G(z|x))' % prefix, color=colors[1], alpha=0.9)
    ax.plot(step, g_recon, label='%s - G(z|x)' % prefix, color=colors[2], alpha=0.9)


def _plot_gan(ax, base_data, new_data, avg=False):
    _plot_gan_axis(ax, base_data, ['r', 'b', 'g'], prefix='w/ random sample', avg=avg)
    _plot_gan_axis(ax, new_data, ['m', 'c', 'y'], prefix='w/o random sample', avg=avg)
    ax.grid()
    ax.set_yscale('log')


def _plot_gans(base_data, new_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    _plot_gan(ax1, base_data.train, new_data.train, avg=True)
    ax1.set(title='Training Loss')
    ax2 = fig.add_subplot(2, 1, 2)
    _plot_gan(ax2, base_data.test, new_data.test)
    ax2.set(xlabel='Steps', ylabel='Log Loss', title='Test Loss')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.33), ncol=2)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    return fig


def _plot_feats(base_data, new_data):
    fig, ax = plt.subplots()
    _plot_envelopped(
        ax, base_data.train['STEP'], base_data.train['F_RECON'],
        label='Input vs Reconstructed - Training w/ random sample', color='b')
    _plot_envelopped(
        ax, new_data.train['STEP'], new_data.train['F_RECON'],
        label='Input vs Reconstructed - Training w/o random sample', color='r')
    ax.plot(
        base_data.test['STEP'], base_data.test['F_RECON'],
        label='Input vs Reconstructed - Test w/ random sample', color='c')
    ax.plot(
        new_data.test['STEP'], new_data.test['F_RECON'],
        label='Input vs Reconstructed - Test w/o random sample', color='m')
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Feature Mathing Error', title='Feature Matching')
    return fig


def _plot_pixels(base_data, new_data):
    fig, ax = plt.subplots()
    _plot_envelopped(
        ax, base_data.train['STEP'], base_data.train['PIXEL'],
        label='Training w/ random sample', color='b')
    _plot_envelopped(
        ax, new_data.train['STEP'], new_data.train['PIXEL'],
        label='Training w/o random sample', color='r')
    ax.plot(
        base_data.test['STEP'], base_data.test['PIXEL'],
        label='Test w/ random sample', color='c')
    ax.plot(
        new_data.test['STEP'], new_data.test['PIXEL'],
        label='Test w/o random sample', color='m')
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Pixel Error', title='Pixel Error')
    return fig


def _main():
    args = _parse_args()
    base_data = _load(args.input[0])
    new_data = _load(args.input[1])
    fig = _plot_klds(base_data, new_data)
    fig.savefig('kld.svg')
    fig = _plot_gans(base_data, new_data)
    fig.savefig('gan.svg')
    fig = _plot_feats(base_data, new_data)
    fig.savefig('feats.svg')
    fig = _plot_pixels(base_data, new_data)
    fig.savefig('pixel.svg')


if __name__ == '__main__':
    _main()
