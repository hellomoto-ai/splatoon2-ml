import logging
import subprocess

import cv2
import imageio
import numpy as np

_LG = logging.getLogger(__name__)


def load_image(path, resize=None):
    """Load image from file into CHW-RGB within [-1.0, 1.0]

    Parameters
    ----------
    resize : tuple of int
        (weight, height)
    """
    img = imageio.imread(path)
    if resize:
        img = cv2.resize(img, resize)
    img = 2 * (img.astype('float') / 255) - 1.0
    # (H, W, C) -> (C, H, W)
    img = img.transpose((2, 0, 1))
    return img


def save_image(img, path):
    """Save image to file"""
    # (C, H, W) -> (H, W, C)
    img = img.transpose((1, 2, 0))
    img = 255 * (img + 1.0) / 2
    img = img.clip(min=0, max=255).astype('uint8')
    imageio.imwrite(path, img)


# TODO: Turn this into class with context manager
def load_video(path, scale, frame_rate=30, debug=False):
    width, height = scale
    command = [
        'ffmpeg',
        '-i', path,
        '-f', 'rawvideo',
        '-r', str(frame_rate),
        '-pix_fmt', 'rgb24',
        '-vf', 'scale=%d:%d' % (width, height),
        '-',
    ]
    frame_size = 3 * width * height
    shape = (height, width, 3)
    process = subprocess.Popen(
        args=command,
        stdout=subprocess.PIPE,
        stderr=None if debug else subprocess.DEVNULL,
        bufsize=32*frame_size,
    )
    n_yield = 0
    while True:
        val = process.stdout.read(frame_size)
        if len(val) != frame_size:
            break
        frame = np.frombuffer(val, dtype='uint8')
        frame = frame.reshape(shape).transpose(2, 0, 1)
        yield frame
        n_yield += 1
    _LG.info('Finished key frame extraction: %d extracted.', n_yield)
    returncode = process.wait(timeout=0.5)
    if returncode != 0:
        _LG.error(
            'Key frame decoding process (ffmpeg) did not '
            'complete correctly. Return code: %s', returncode
        )


# TODO: Turn this into class with context manager
def save_video(path, scale, frame_rate=30, debug=False):
    width, height = scale
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-framerate', str(frame_rate),
        '-s', '%dx%d' % (width, height),
        '-pix_fmt', 'rgb24',
        '-i', '-',
        '-an',
        '-codec:v', 'libx265',
        '-preset', 'veryslow',
    ]
    if width % 2 == 0 and height % 2 == 0:
        command += ['-pix_fmt', 'yuv420p']
    command += [path]
    frame_size = 3 * width * height
    process = subprocess.Popen(
        args=command,
        stdin=subprocess.PIPE,
        stdout=None if debug else subprocess.DEVNULL,
        stderr=None if debug else subprocess.DEVNULL,
        bufsize=frame_size,
    )
    return process
