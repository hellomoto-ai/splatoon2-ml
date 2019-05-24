#!/usr/bin/env python
import os
from collections import namedtuple

import pandas
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

# https://stackoverflow.com/a/55652330/3670924
COLORS = matplotlib.cm.get_cmap('tab20').colors

TARGET_KLD = [0.05, 0.1, 0.2, 0.5]

LABELS = ['Target KL: %s' % v for v in TARGET_KLD]

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def _get_result_dir(exp):
    return os.path.join(BASE_DIR, 'results', exp, 'result.csv')


def _parse_args():
    import argparse

    results = [
        _get_result_dir('2019-05-13-16-1585695'),
        _get_result_dir('2019-05-11-18-48b3a7f'),
        _get_result_dir('2019-05-13-22-31f766f'),
        _get_result_dir('2019-05-13-00-4c2d253'),
    ]
    output_dir = os.path.join(BASE_DIR, 'assets', '2019-05-30')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i', nargs=3,
        default=results,
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=output_dir,
    )
    return parser.parse_args()


###############################################################################
def _get_ax(color=COLORS):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=color)
    return fig, ax


###############################################################################
Data = namedtuple('Data', ['train', 'test'])


def _subsample(data, n, func=np.mean):
    return np.asarray([func(data[i:i+n]) for i in range(0, len(data), n)])


def _load(path):
    data = pandas.read_csv(path)
    train_data = data[data['PHASE'] == 'train']
    test_data = data[data['PHASE'] == 'test']
    return Data(train_data, test_data)


###############################################################################
def _plot_klds(data):
    fig, ax = _get_ax()

    for i, datum in enumerate(data):
        ax.axhline(
            TARGET_KLD[i], color=COLORS[i*2+1],
            linestyle='--', linewidth=1.5, label='_nolegend_')
        label = '%s - Train' % LABELS[i]
        _plot_envelopped(
            ax, datum.train['STEP'], datum.train['KLD'],
            label=label, color=COLORS[i*2 + 1])
        label = '%s - Test' % LABELS[i]
        ax.plot(
            datum.test['STEP'], datum.test['KLD'],
            label=label, color=COLORS[i*2])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='KLD', title='KL Divergence')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_beta(data):
    fig, ax = _get_ax()
    for i, datum in enumerate(data):
        label = LABELS[i]
        ax.plot(
            datum.train['STEP'], datum.train['BETA'],
            label=label, color=COLORS[i*2])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='$\\beta$', title='Trend of $\\beta$')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_envelopped(ax, x, y, label, color=None, max_samples=50):
    n = len(x) // max_samples
    x = _subsample(x, n)
    y_max = _subsample(y, n, np.max)
    y_min = _subsample(y, n, np.min)
    y_mean = _subsample(y, n)
    ax.fill_between(
        x, y_min, y_max, color=color, label='_nolegend_', alpha=0.25)
    ax.plot(x, y_mean, label=label, color=color)


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
        _plot_envelopped(ax, step, d_fake, label='D(G(z))', color=COLORS[4])
        _plot_envelopped(ax, step, g_recon, label='G(z|x)', color=COLORS[6])
        _plot_envelopped(ax, step, g_fake, label='G(z)', color=COLORS[8])
    else:
        ax.plot(step, real, label='D(x)', color=COLORS[0])
        ax.plot(step, d_recon, label='D(G(z|x))', color=COLORS[2])
        ax.plot(step, d_fake, label='D(G(z))', color=COLORS[4])
        ax.plot(step, g_recon, label='G(z|x)', color=COLORS[6])
        ax.plot(step, g_fake, label='G(z)', color=COLORS[8])


def _plot_gan(ax, data, envelop=False):
    _plot_gan_axis(ax, data, envelop)
    ax.grid()
    ax.set_yscale('log')


def _plot_gans(data):
    n_col = len(LABELS)
    fig = plt.figure(figsize=[6.4 * n_col, 7.2])
    axes = [fig.add_subplot(2, n_col, i) for i in range(1, 1+2*n_col)]
    for i, datum in enumerate(data):
        ax = axes[i]
        _plot_gan(ax, datum.train, envelop=True)
        ax.set(ylim=[0.8e-3, 5e1])
        ax = axes[i+n_col]
        _plot_gan(ax, datum.test)
        ax.set(ylim=[1.0e-3, 4e1])
    fig.suptitle('GAN Loss')
    axes[2].legend(loc='upper center', bbox_to_anchor=(-0.7, 1.2), ncol=5)
    return fig


###############################################################################
def _plot_feats(data):
    fig, ax = _get_ax()
    for i, datum in enumerate(data):
        label = '%s - Train' % LABELS[i]
        _plot_envelopped(
            ax, datum.train['STEP'], datum.train['F_RECON'],
            label=label, color=COLORS[2*i+1])
        label = '%s - Test' % LABELS[i]
        ax.plot(
            datum.test['STEP'], datum.test['F_RECON'],
            label=label, color=COLORS[2*i])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Feature Mathing Error', title='Feature Matching')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_pixels(data):
    fig, ax = _get_ax()
    for i, datum in enumerate(data):
        label = '%s - Train' % LABELS[i]
        _plot_envelopped(
            ax, datum.train['STEP'], datum.train['PIXEL'], label=label, color=COLORS[2*i+1])
        label = '%s - Test' % LABELS[i]
        ax.plot(
            datum.test['STEP'], datum.test['PIXEL'], label=label, color=COLORS[2*i])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Pixel Error', title='Pixel Error')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_latent(ax, data, color, label, max_samples=30):
    if 'Z_MEAN' in data:
        steps, z_min, z_max = data['STEP'], data['Z_MIN'], data['Z_MAX']
        z_mean, z_var = data['Z_MEAN'], data['Z_VAR']
        if len(steps) > max_samples:
            n = len(steps) // max_samples
            steps = _subsample(steps, n)
            z_min = _subsample(z_min, n, np.min)
            z_max = _subsample(z_max, n, np.max)
            z_mean = _subsample(z_mean, n)
            z_var = _subsample(z_var, n)
        ax.fill_between(steps, z_min, z_max, color=color, alpha=0.3)
        ax.errorbar(steps, z_mean, yerr=z_var, label=label, color=color)
        ax.grid()
        ax.legend()


def _plot_latent_stats(data):
    n_col = len(LABELS)
    fig = plt.figure(figsize=[6.4 * n_col, 7.2])
    ax = [fig.add_subplot(2, n_col, i) for i in range(1, 1+2*n_col)]
    for i, datum in enumerate(data):
        label = '%s - Train' % LABELS[i]
        _plot_latent(ax[i], datum.train, color=COLORS[2*i], label=label)
        ax[i].set(ylim=[-8, 8])
        label = '%s - Test' % LABELS[i]
        _plot_latent(ax[i+n_col], datum.test, color=COLORS[2*i], label=label)
        ax[i+3].set(ylim=[-8, 8])
    fig.suptitle('Encoded sample distribution (Min/Mean/Var/Max)')
    return fig


###############################################################################
def _main():
    args = _parse_args()
    data = [_load(f) for f in args.input]
    fig = _plot_klds(data)
    fig.savefig(os.path.join(args.output_dir, 'kld.svg'))
    fig = _plot_pixels(data)
    fig.savefig(os.path.join(args.output_dir, 'pixel.svg'))
    fig = _plot_beta(data)
    fig.savefig(os.path.join(args.output_dir, 'beta.svg'))
    fig = _plot_latent_stats(data)
    fig.savefig(os.path.join(args.output_dir, 'latent.svg'))
    fig = _plot_gans(data)
    fig.savefig(os.path.join(args.output_dir, 'gan.svg'))
    fig = _plot_feats(data)
    fig.savefig(os.path.join(args.output_dir, 'feats.svg'))


if __name__ == '__main__':
    _main()