#!/usr/bin/env python
import os
from collections import namedtuple

import pandas
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

# https://stackoverflow.com/a/55652330/3670924
COLORS = matplotlib.cm.get_cmap('tab20').colors


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i',
        default=os.path.join('results', '2019-05-06-01-dd01034', 'result.csv'),
        help='CSV files. Base result and new results.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=os.path.join('assets', '2019-05-16'),
    )
    return parser.parse_args()


###############################################################################
def _get_ax(color=COLORS):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=color)
    return fig, ax


def _plot_envelopped(ax, x, y, label, color=None, n=100):
    ax.plot(x, y, label='_nolegend_', color=color, alpha=0.25)
    ax.plot(_avg(x, n), _avg(y, n), label=label, color=color)


###############################################################################
Data = namedtuple('Data', ['train', 'test'])


def _avg(data, n):
    return np.asarray([np.mean(data[i:i+n]) for i in range(0, len(data), n)])


def _load(path):
    data = pandas.read_csv(path)
    train_data = data[data['PHASE'] == 'train']
    test_data = data[data['PHASE'] == 'test']
    return Data(train_data, test_data)


###############################################################################
def _plot_klds(data):
    fig, ax = _get_ax()
    ax.plot(
        data.train['STEP'], data.train['KLD'], label='Train', color=COLORS[0])
    ax.plot(
        data.test['STEP'], data.test['KLD'], label='Test', color=COLORS[2])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='KLD', title='KL Divergence')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_gan_axis(ax, data, envelop):
    step = data['STEP']
    real = data['D_REAL']
    d_recon = data['D_RECON']
    d_fake = data['D_FAKE']
    g_recon = data['G_RECON']
    g_fake = data['G_FAKE']
    if envelop:
        _plot_envelopped(ax, step, real, label='D(x)', color=COLORS[0])
        _plot_envelopped(ax, step, d_recon, label='D(G(z|x))', color=COLORS[2])
        _plot_envelopped(ax, step, g_recon, label='G(z|x)', color=COLORS[4])
        _plot_envelopped(ax, step, d_fake, label='D(G(z))', color=COLORS[6])
        _plot_envelopped(ax, step, g_fake, label='G(z)', color=COLORS[8])
    else:
        ax.plot(step, real, label='D(x)', color=COLORS[0], alpha=0.9)
        ax.plot(step, d_recon, label='D(G(z|x))', color=COLORS[2], alpha=0.9)
        ax.plot(step, g_recon, label='G(z|x)', color=COLORS[4], alpha=0.9)
        ax.plot(step, d_fake, label='D(G(z))', color=COLORS[6], alpha=0.9)
        ax.plot(step, g_fake, label='G(z)', color=COLORS[8], alpha=0.9)


def _plot_gan(ax, data, envelop=False):
    _plot_gan_axis(ax, data, envelop)
    ax.grid()
    ax.set_yscale('log')


def _plot_gans(data):
    fig = plt.figure(figsize=[6.4, 7.2])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    _plot_gan(ax1, data.train, envelop=True)
    _plot_gan(ax2, data.test)
    ax1.set(title='Training Loss', ylabel='Log Loss', ylim=[1e-2, 1e1])
    ax2.set(xlabel='Steps', ylabel='Log Loss', title='Test Loss')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    return fig


###############################################################################
def _plot_feats(data):
    fig, ax = _get_ax()
    _plot_envelopped(
        ax, data.train['STEP'], data.train['F_RECON'],
        label='Train', color=COLORS[0])
    ax.plot(
        data.test['STEP'], data.test['F_RECON'],
        label='Test', color=COLORS[2])
    ax.grid()
    ax.legend()
    ax.set(
        xlabel='Steps', ylabel='Feature Mathing Error', title='Feature Matching')
    return fig


###############################################################################
def _plot_pixels(data):
    fig, ax = _get_ax()
    ax.plot(
        data.test['STEP'], data.test['PIXEL'], label='Test', color=COLORS[2])
    _plot_envelopped(
        ax, data.train['STEP'], data.train['PIXEL'],
        label='Train', color=COLORS[0])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Pixel Error', title='Pixel Error')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_latent_stats(data):
    fig, ax = _get_ax(COLORS[::2])
    beta = 1
    i = 0
    label = 'β=%s - Test' % beta

    if 'Z_DIST_MIN' in data.test:
        ax.errorbar(
            data.test['STEP'], data.test['Z_DIST_MEAN'],
            yerr=np.vstack((data.test['Z_DIST_MIN'], data.test['Z_DIST_MAX'])),
            label=label, color=COLORS[2*i])
        # ax.fill_between(
        #     data.test['STEP'], data.test['Z_MEAN_MIN'], data.test['Z_MEAN_MAX'])
    ax.grid()
    ax.legend()
    ax.set(
        xlabel='Steps', ylabel='Euclidean Distance from Origin',
        title='Min/Mean/Max Euclidean distance of estimated latent point from Origin'
    )
    ax.set_yscale('log')
    return fig


###############################################################################
def _main():
    args = _parse_args()
    data = _load(args.input)
    fig = _plot_klds(data)
    fig.savefig(os.path.join(args.output_dir, 'kld.svg'))
    fig = _plot_gans(data)
    fig.savefig(os.path.join(args.output_dir, 'gan.svg'))
    fig = _plot_feats(data)
    fig.savefig(os.path.join(args.output_dir, 'feats.svg'))
    fig = _plot_pixels(data)
    fig.savefig(os.path.join(args.output_dir, 'pixel.svg'))
    fig = _plot_latent_stats(data)
    fig.savefig(os.path.join(args.output_dir, 'latent.svg'))


if __name__ == '__main__':
    _main()
