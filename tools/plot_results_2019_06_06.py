#!/usr/bin/env python
import os
from collections import namedtuple

import pandas
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

# https://stackoverflow.com/a/55652330/3670924
COLORS = matplotlib.cm.get_cmap('tab20').colors


###############################################################################
def _get_ax(color=COLORS):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=color)
    return fig, ax


###############################################################################
def _plot_klds(data, labels):
    fig, ax = _get_ax()
    for i, datum in enumerate(data):
        label = '%s - Train' % labels[i]
        _plot_envelopped(
            ax, datum.train['STEP'], datum.train['KLD'],
            label=label, color=COLORS[2*i+1])
        label = '%s - Test' % labels[i]
        ax.plot(
            datum.test['STEP'], datum.test['KLD'],
            label=label, color=COLORS[2*i])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='KLD', title='KL Divergence')
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_beta(data, labels):
    fig, ax = _get_ax()
    for i, datum in enumerate(data):
        ax.plot(
            datum.train['STEP'], datum.train['BETA'],
            label=labels[i], color=COLORS[2*i])
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
    fig = plt.figure(figsize=[6.4, 7.2])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    _plot_gan(ax1, data.train, envelop=True)
    _plot_gan(ax2, data.test)
    ax1.set(title='Training Loss')  # , ylim=[8e-4, 8e1])
    ax2.set(xlabel='Steps', ylabel='Log Loss', title='Test Loss')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.33), ncol=5, fontsize=6)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    return fig


###############################################################################
def _plot_feats(data, labels):
    fig, ax = _get_ax()
    for i, datum in enumerate(data):
        _plot_envelopped(
            ax, datum.train['STEP'], datum.train['F_RECON'],
            label=labels[i], color=COLORS[2*i+1])
        ax.plot(
            datum.test['STEP'], datum.test['F_RECON'],
            label=labels[i], color=COLORS[2*i])
    ax.grid()
    ax.legend()
    ax.set(
        xlabel='Steps', ylabel='Feature Mathing Error',
        title='Feature Matching Error', ylim=[1e-1, 1e3])
    ax.set_yscale('log')
    return fig


###############################################################################
def _plot_pixels(data, labels):
    fig, ax = _get_ax()

    for i, datum in enumerate(data):
        label = '%s - Train' % labels[i]
        _plot_envelopped(
            ax, datum.train['STEP'], datum.train['PIXEL'],
            label=label, color=COLORS[i*2+1])
        label = '%s - Test' % labels[i]
        ax.plot(
            datum.test['STEP'], datum.test['PIXEL'],
            label=label, color=COLORS[i*2])
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Pixel Error', title='Pixel Error')
    ax.set_yscale('log')
    return fig


###############################################################################
def _subsample(data, n, func=np.mean):
    return np.asarray([func(data[i:i+n]) for i in range(0, len(data), n)])


def _plot_envelopped2(ax, x, y, var, label, color=None, max_samples=50):
    n = len(x) // max_samples
    x = _subsample(x, n)
    y_mean = _subsample(y, n)
    y_std = _subsample(np.sqrt(var), n)
    y_min = (y_mean - y_std).clip(min=0)
    y_max = y_mean + y_std
    ax.fill_between(
        x, y_min, y_max, color=color, label='_nolegend_', alpha=0.25)
    ax.plot(x, y_mean, label=label, color=color)


def _plot_latent_stats(data, labels):
    fig = plt.figure(figsize=[6.4, 10.8])
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    n = len(data)
    for i, (data_, label) in enumerate(zip(data[::-1], labels[::-1])):
        datum = data_.train
        color = COLORS[2*(n - 1 - i)]
        _plot_envelopped2(
            ax1, datum['STEP'], datum['Z_MEAN'], datum['Z_VAR'],
            color=color, label=label)
        ax = ax2 if i == n - 1 else ax3
        _plot_envelopped2(
            ax, datum['STEP'], datum['Z_STD_MEAN'], datum['Z_STD_VAR'],
            color=color, label=label)

    ax1.legend()
    ax1.grid()
    ax1.set(title='Distribution of Z_MEAN')
    ax2.legend()
    ax2.grid()
    ax2.set(title='Distribution of Z_STDDEV')
    ax3.legend()
    ax3.grid()
    ax3.set(xlabel='Steps', ylim=[0, 0.1])
    return fig


###############################################################################
def _plot(data, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig = _plot_klds(data, labels)
    fig.savefig(os.path.join(output_dir, 'kld.svg'))
    fig = _plot_beta(data, labels)
    fig.savefig(os.path.join(output_dir, 'beta.svg'))
    '''
    fig = _plot_gans(data)
    fig.savefig(os.path.join(args.output_dir, 'gan.svg'))
    '''
    fig = _plot_feats(data, labels)
    fig.savefig(os.path.join(output_dir, 'feats.svg'))
    fig = _plot_pixels(data, labels)
    fig.savefig(os.path.join(output_dir, 'pixel.svg'))
    fig = _plot_latent_stats(data, labels)
    fig.savefig(os.path.join(output_dir, 'latent.svg'))


###############################################################################
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


Data = namedtuple('Data', ['train', 'test'])


def _load(path):
    data = pandas.read_csv(path)
    train_data = data[data['PHASE'] == 'train']
    test_data = data[data['PHASE'] == 'test']
    return Data(train_data, test_data)


def _get_result_file(exp):
    return os.path.join(BASE_DIR, 'results', exp, 'result.csv')


def _plot_exp2():
    labels = [
        'Single Point KLD',
        'Batch: Momentum = 0.0',
        'Batch: Momentum = 0.1',
        'Batch: Momentum = 0.9',
    ]
    data = [
        _load(_get_result_file(exp)) for exp in [
            '2019-05-24-04-2e5d3bf',  # base
            '2019-05-22-21-9a539af',  # batch KLD: Momentum 0.0
            '2019-05-23-13-872495d',  # batch KLD: Momentum 0.1
            '2019-05-22-06-d0359c8',  # batch KLD: Momentum 0.9
        ]
    ]
    output_dir = os.path.join(BASE_DIR, 'assets', '2019-06-06', 'exp2')
    _plot(data, labels, output_dir)


def _plot_exp1():
    labels = [
        'Single Point',
        'Batch: Momentum = 0.0',
        'Batch: Momentum = 0.1',
        'Batch: Momentum = 0.9',
    ]
    data = [
        _load(_get_result_file(exp)) for exp in [
            '2019-05-24-19-c1c26d9',  # base
            '2019-05-26-05-48e028f',  # batch KLD: Momentum 0.0
            '2019-05-29-17-4bd24ee',  # batch KLD: Momentum 0.1
            '2019-05-25-13-cfe6e46',  # batch KLD: Momentum 0.9
        ]
    ]
    output_dir = os.path.join(BASE_DIR, 'assets', '2019-06-06', 'exp1')

    # Fix: logvar -> std
    for datum in data[1:]:
        for key in ['Z_STD_MEAN', 'Z_STD_MIN', 'Z_STD_MAX', 'Z_STD_VAR']:
            datum.test[key] = np.exp(0.5 * datum.test[key])
            datum.train[key] = np.exp(0.5 * datum.train[key])
    _plot(data, labels, output_dir)


def _main():
    _plot_exp1()
    _plot_exp2()


if __name__ == '__main__':
    _main()
