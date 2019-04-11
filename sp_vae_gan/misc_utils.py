import csv
import collections


class MeanTracker(collections.defaultdict):
    """Keep track of mean values"""
    def __init__(self):
        super().__init__(float)
        self.step = 1

    def update(self, data):
        for key, val in data.items():
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
