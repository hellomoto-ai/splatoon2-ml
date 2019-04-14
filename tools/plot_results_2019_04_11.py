#!/usr/bin/env python
import pandas
import numpy as np
import matplotlib.pyplot as plt


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    return parser.parse_args()


def _split(data):
    train_data = data[data['PHASE'] == 'train']
    test_data = data[data['PHASE'] == 'test']
    return train_data, test_data


def _avg(data, n=100):
    return [np.mean(data[i:i+n]) for i in range(0, len(data), n)]


def _plot_envelopped(ax, x, y, label, color, linestyle='-'):
    ax.plot(x, y, label='_nolegend_', color=color, alpha=0.25)
    ax.plot(_avg(x), _avg(y), label=label, color=color, linestyle=linestyle)


def _plot_kld(train_data, test_data):
    fig, ax = plt.subplots()
    _plot_envelopped(ax, train_data['STEP'], train_data['KLD'], label='Training', color='b')
    ax.plot(test_data['STEP'], test_data['KLD'], label='Test', color='c')
    ax.grid()
    ax.legend()
    ax.set_ylim(0, 12)
    ax.set(xlabel='Steps', ylabel='KLD', title='KL Divergence')
    return fig


def _plot_gan_axis(ax, data, envelop=False, fake=True):
    step = data['STEP']
    if envelop:
        _plot_envelopped(ax, step, data['D_REAL'], label='D(x)', color='r')
        _plot_envelopped(ax, step, data['D_RECON'], label='D(G(z|x))', color='b')
        _plot_envelopped(ax, step, data['G_RECON'], label='G(z|x)', color='m')
        if fake:
            _plot_envelopped(ax, step, data['D_FAKE'], label='D(G(z))', color='g')
            _plot_envelopped(ax, step, data['G_FAKE'], label='G(z)', color='y')
    else:
        ax.plot(step, data['D_REAL'], label='D(x)', color='r')
        ax.plot(step, data['D_RECON'], label='D(G(z|x))', color='b')
        ax.plot(step, data['G_RECON'], label='G(z|x)', color='m')
        if fake:
            ax.plot(step, data['D_FAKE'], label='D(G(z))', color='g')
            ax.plot(step, data['G_FAKE'], label='G(z)', color='y')
    ax.grid()
    ax.set_yscale('log')


def _plot_gan(train_data, test_data, fake=True, ylim=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    _plot_gan_axis(ax1, train_data, envelop=True, fake=fake)
    ax1.set(ylabel='Log Loss', title='Training Loss')
    if ylim:
        ax1.set_ylim(*ylim)
    ax2 = fig.add_subplot(2, 1, 2)
    _plot_gan_axis(ax2, test_data, fake=fake)
    ax2.set(xlabel='Steps', ylabel='Log Loss', title='Test Loss')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.33), ncol=5)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    return fig


def _plot_feats(train_data, test_data):
    fig, ax = plt.subplots()
    _plot_envelopped(
        ax, train_data['STEP'], train_data['F_FAKE'],
        label='Input vs Fake - Training', color='b')
    ax.plot(
        test_data['STEP'], test_data['F_FAKE'],
        label='Input vs Fake - Test', color='c')
    _plot_envelopped(
        ax, train_data['STEP'], train_data['F_RECON'],
        label='Input vs Reconstructed - Training', color='r')
    ax.plot(
        test_data['STEP'], test_data['F_RECON'],
        label='Input vs Reconstructed - Test', color='m')
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Feature Mathing Error', title='Feature Matching')
    return fig


def _plot_pixel(train_data, test_data):
    fig, ax = plt.subplots()
    _plot_envelopped(
        ax, train_data['STEP'], train_data['PIXEL'],
        label='Training', color='b')
    ax.plot(
        test_data['STEP'], test_data['PIXEL'],
        label='Test', color='c')
    ax.grid()
    ax.legend()
    ax.set(xlabel='Steps', ylabel='Pixel Error', title='Pixel Error')
    return fig


def _main():
    args = _parse_args()
    data = pandas.read_csv(args.input)
    train_data, test_data = _split(data)
    fig = _plot_kld(train_data, test_data)
    fig.savefig('kld.svg')
    fig = _plot_gan(train_data, test_data)
    fig.savefig('gan.svg')
    fig = _plot_gan(train_data, test_data, fake=False, ylim=(1e-4, 1e2))
    fig.savefig('gan_without_fake.svg')
    fig = _plot_feats(train_data, test_data)
    fig.savefig('feats.svg')
    fig = _plot_pixel(train_data, test_data)
    fig.savefig('pixel.svg')


if __name__ == '__main__':
    _main()
