import csv
import collections

import numpy as np


class StatsTracker(collections.defaultdict):
    """Keep track of mean values"""
    def __init__(self):
        super().__init__(float)
        self.step = 1

    def update(self, data):
        for key, val in data.items():
            if key.endswith('_min'):
                val = np.min(val)
                self[key] = min(self.get(key, val), val)
            elif key.endswith('_max'):
                val = np.max(val)
                self[key] = max(self.get(key, val), val)
            else:
                val = np.mean(val)
                self[key] += (val - self[key]) / self.step
        self.step += 1


class CSVWriter:
    """CSV Writer"""
    def __init__(self, fields, fileobj):
        self.fileobj = fileobj
        self.writer = csv.DictWriter(fileobj, fieldnames=fields)
        self.writer.writeheader()

    def write(self, **kwargs):
        self.writer.writerow(kwargs)
        self.fileobj.flush()
