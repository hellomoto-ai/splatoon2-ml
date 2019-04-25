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
        '--input', '-i', required=True, nargs=5,
        help='CSV files. Base result and new results.'
    )
    parser.add_argument(
        '--output-dir', '-o', required=True,
    )
    return parser.parse_args()


###############################################################################
def _get_ax(color=COLORS):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=color)
    return fig, ax


###############################################################################
Data = namedtuple('Data', ['train', 'test'])


def _avg(data, n=500):
    return np.asarray([np.mean(data[i:i+n]) for i in range(0, len(data), n)])


def _load(path, n=500):
    data = pandas.read_csv(path)
    train_data = data[data['PHASE'] == 'train']
    for key in train_data:
        values = train_data[key].values
        if key != 'PHASE':
            values = _avg(values, n)
        else:
            values = values[::n]
        train_data[key] = pandas.Series(values)
    test_data = data[data['PHASE'] == 'test']
    return Data(train_data, test_data)


###############################################################################
def _plot_klds(data_set):
    fig, ax = _get_ax()
    for i, data in enumerate(data_set):
        beta = 2 ** (i - 1) if i else 0.1
        label = 'β=%s - Test' % beta
        ax.plot(data.test['STEP'], data.test['KLD'], label=label)
        label = 'β=%s - Train' % beta
        ax.plot(data.train['STEP'], data.train['KLD'], label=label, alpha=0.5)
    ax.grid()
    ax.legend()
    # ax.set_ylim(0, 25)
    ax.set(xlabel='Steps', ylabel='KLD', title='KL Divergence')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_gan_axis(ax, data, colors, prefix):
    step = data['STEP']
    real = data['D_REAL']
    d_recon = data['D_RECON']
    g_recon = data['G_RECON']
    ax.plot(step, real, label='%s - D(x)' % prefix, color=colors[0], alpha=0.9)
    ax.plot(step, d_recon, label='%s - D(G(z|x))' % prefix, color=colors[1], alpha=0.9)
    ax.plot(step, g_recon, label='%s - G(z|x)' % prefix, color=colors[2], alpha=0.9)


def _plot_gan(ax, data, colors, prefix):
    _plot_gan_axis(ax, data, colors, prefix=prefix)
    ax.grid()
    ax.set_yscale('log')


def _plot_gans(data_set):
    colors = matplotlib.cm.get_cmap('tab20c').colors
    fig = plt.figure(figsize=[6.4, 7.2])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for i, data in enumerate(data_set):
        colors_ = colors[4*i:4*i+3]
        beta = 2 ** (i - 1) if i else 0.1
        prefix = 'β=%s' % beta
        _plot_gan(ax1, data.train, colors_, prefix)
        _plot_gan(ax2, data.test, colors_, prefix)
    ax1.set(title='Training Loss')
    ax2.set(xlabel='Steps', ylabel='Log Loss', title='Test Loss')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.33), ncol=5, fontsize=6)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    return fig


###############################################################################
def _plot_feats(data_set):
    fig, ax = _get_ax()
    for i, data in enumerate(data_set):
        beta = 2 ** (i - 1) if i else 0.1
        label = 'β=%s - Test' % beta
        ax.plot(data.test['STEP'], data.test['F_RECON'], label=label)
        label = 'β=%s - Train' % beta
        ax.plot(data.train['STEP'], data.train['F_RECON'], label=label, alpha=0.5)
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Feature Mathing Error', title='Feature Matching')
    return fig


###############################################################################
def _plot_pixels(data_set):
    fig, ax = _get_ax()
    for i, data in enumerate(data_set):
        beta = 2 ** (i - 1) if i else 0.1
        label = 'β=%s - Test' % beta
        ax.plot(data.test['STEP'], data.test['PIXEL'], label=label)
        label = 'β=%s - Train' % beta
        ax.plot(data.train['STEP'], data.train['PIXEL'], label=label, alpha=0.5)
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Pixel Error', title='Pixel Error')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_latent_stats(data_set):
    fig, ax = _get_ax(COLORS[::2])
    for i, data in enumerate(data_set):
        beta = 2 ** (i - 1) if i else 0.1
        label = 'β=%s - Test' % beta

        if 'Z_MEAN_MIN' in data.test:
            ax.errorbar(
                data.test['STEP'], data.test['Z_MEAN'],
                yerr=data.test['Z_VAR'],
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
    data = [_load(f) for f in args.input]
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
