# Copyright (c) Meta Platforms, Inc. and affiliates.

import csv
import re

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=' ', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RunningStats(object):
    """Computes running mean and standard deviation
    Adapted from https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, n=0.0, m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def clear(self):
        self.n = 0.0

    def push(self, x, per_dim=True):
        x = np.array(x).copy().astype('float32')
        # process input
        if per_dim:
            self.update_params(x)
        else:
            for el in x.flatten():
                self.update_params(el)

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.0
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            sum_ns = self.n + other.n
            prod_ns = self.n * other.n
            delta2 = (other.m - self.m) ** 2.0
            return RunningStats(
                sum_ns,
                (self.m * self.n + other.m * other.n) / sum_ns,
                self.s + other.s + delta2 * prod_ns / sum_ns,
            )
        else:
            self.push(other)
            return self

    @property
    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n) if self.n else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance())

    def __repr__(self):
        return (
            '<RunningMean(mean={: 2.4f}, std={: 2.4f}, n={: 2f}, m={: 2.4f}, s={: 2.4f})>'.format(
                self.mean, self.std, self.n, self.m, self.s
            )
        )

    def __str__(self):
        return 'mean={: 2.4f}, std={: 2.4f}'.format(self.mean, self.std)


class RunningStatsWithBuffer(RunningStats):
    def __init__(self, path=None, row_id_map=None, data=None, header=None, n=0.0, m=None, s=None):
        super(RunningStatsWithBuffer, self).__init__(n, m, s)
        self.path = path

        if data is None:
            self.data = []
        else:
            assert isinstance(data, list) and any(isinstance(i, list) for i in data)
            self.data = data

        if row_id_map is None:
            self.row_id_map = {}
        else:
            assert isinstance(row_id_map, dict)
            self.row_id_map = row_id_map

        if header is None:
            self.header = None
        else:
            assert isinstance(header, list)
            self.header = header

    def push(self, id, value, per_dim=True):
        if id in self.row_id_map:
            return
        self.row_id_map[id] = len(self.data)
        self.data.append(value if isinstance(value, list) else [value])
        super(RunningStatsWithBuffer, self).push(value)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            for k, v in other.row_id_map.items():
                if k in self.row_id_map:
                    continue
                self.row_id_map[k] = len(self.data)
                self.data.append(other.data[v])

            data_array = np.array(self.data).copy().astype('float32')
            return RunningStatsWithBuffer(
                self.path,
                self.row_id_map,
                self.data,
                self.header,
                len(self.data),
                np.nanmean(data_array, 0),
                np.nanvar(data_array, 0),
            )
        else:
            self.push(*other)
            return self

    def dump(self):
        def natural_sort(l):
            def convert(text):
                return int(text) if text.isdigit() else text.lower()

            return sorted(l, key=lambda key: [convert(c) for c in re.split('([0-9]+)', key[0])])

        table = [self.header]
        table.extend([[k] + self.data[v] for k, v in self.row_id_map.items()])
        table[1:] = natural_sort(table[1:])

        with open(self.path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(table)

    @property
    def mean(self):
        data_array = np.array(self.data).copy().astype('float32')
        return np.nanmean(data_array, 0)

    def variance(self):
        data_array = np.array(self.data).copy().astype('float32')
        return np.nanvar(data_array, 0)
