import os

import torch.utils.data

from sp_vae_gan import image_util


def _load_frames(flist, root_dir):
    ret = []
    with open(flist, 'r') as fileobj:
        for line in fileobj:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            path, class_ = line.split('\t')
            ret.append((os.path.join(root_dir, path), int(class_)))
    return ret


class Dataset(torch.utils.data.Dataset):
    """Image dataset.

    Parameters
    ----------
    flist : str
        Path to a file which contains the list image paths, relative to
        `root_dir`

    root_dir : str
        Path to the directory where image files are located.

    scale : tuple of int (width, height)
        Rescale image.
    """
    def __init__(self, flist, root_dir, scale):
        self.flist = flist
        self.root_dir = root_dir
        self.frames = _load_frames(flist, root_dir)
        self.scale = scale

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path, class_ = self.frames[idx]
        image = image_util.load_image(path, self.scale)
        return {'image': image, 'class': class_, 'path': path}


def get_dataloader(flist, data_dir, batch_size, scale, shuffle=True):
    return torch.utils.data.DataLoader(
        Dataset(flist, data_dir, scale=scale),
        batch_size=batch_size, shuffle=shuffle, num_workers=4)
