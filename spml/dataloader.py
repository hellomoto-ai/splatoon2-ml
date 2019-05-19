import os
import logging

import numpy as np
import torch.utils.data

from spml import image_util

_LG = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Image dataset.

    Parameters
    ----------
    flist : str
        list image paths, relative to root_dir`

    root_dir : str
        Path to the directory where image files are located.

    scale : tuple of int (width, height)
        Rescale image.
    """
    def __init__(self, flist, root_dir, scale):
        self.flist = flist
        self.root_dir = root_dir
        self.scale = scale

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        path = self.flist[idx]
        full_path = os.path.join(self.root_dir, path)
        image = image_util.load_image(full_path, self.scale)
        return {'image': image, 'path': path}


def _encode(buffer_):
    batch = 2 * np.asarray(buffer_, dtype='float') / 255 - 1.0
    return torch.from_numpy(batch)


class VideoSlicer:
    """Extract frames from a single video"""
    def __init__(
            self, input_file, batch_size=32,
            frame_rate=30, frame_dim=(121, 65), debug=False):
        self.input_file = input_file
        self.batch_size = batch_size
        self.frame_rate = frame_rate
        self.frame_width, self.frame_height = frame_dim
        self.debug = debug

    def __iter__(self):
        loader = image_util.VideoLoader(
            path=self.input_file,
            scale=(self.frame_width, self.frame_height),
            frame_rate=self.frame_rate, debug=self.debug
        )
        buffer_ = []
        with loader:
            for frame in loader:
                buffer_.append(frame)
                if len(buffer_) >= self.batch_size:
                    yield _encode(buffer_)
                    buffer_ = []
            if buffer_:
                yield _encode(buffer_)


class MultiVideoDataLoader:
    """Extract frames from multiple videos

    Parameters
    ----------
    flist : list of dict
        `path` and `duration` key is expected.

    root_dir : str

    frame_rate : int
    """
    def __init__(
            self, flist, root_dir, batch_size=64,
            frame_dim=(121, 65), frame_rate=30,
            randomize=True, random_seed=None, debug=False):
        self.flist = flist
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.frame_rate = frame_rate
        self.frame_width, self.frame_height = frame_dim
        self.randomize = randomize
        self._rng = np.random.RandomState(random_seed)
        self.debug = debug

    def __len__(self):
        return len(self.flist)

    def __iter__(self):
        if self.randomize:
            self._rng.shuffle(self.flist)

        for item in self.flist:
            yield from self._iter_item(item)

    def _iter_item(self, item):
        path, duration = item['path'], item['duration']
        path = os.path.join(self.root_dir, path)

        start = max(0, self._rng.random_sample() * (duration - 5))
        loader = image_util.VideoLoader(
            path=path, seek=start,
            scale=(self.frame_width, self.frame_height),
            frame_rate=self.frame_rate, debug=self.debug
        )
        buffer_ = []
        with loader:
            for frame in loader:
                buffer_.append(frame)
                if len(buffer_) >= self.batch_size:
                    break
            loader.kill()
        if not buffer_:
            _LG.warning('No frame was decoded from %s', item['path'])
            return
        frames = _encode(buffer_)
        yield {'frames': frames, 'path': item['path'], 'start': start}


def get_dataloader(flist, data_dir, batch_size, scale, shuffle=True):
    return torch.utils.data.DataLoader(
        Dataset(flist, data_dir, scale=scale),
        batch_size=batch_size, shuffle=shuffle, num_workers=1)
