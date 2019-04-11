import cv2
import imageio


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
