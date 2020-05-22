import operator
import pathlib
import re
from functools import reduce

import numpy as np


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def get_latest_file_iteration(folder, pattern='*'):
    folder = pathlib.Path(folder)
    matches = [(fp, re.findall(r'\d+', fp.stem)) for fp in folder.glob(pattern)]
    file_itr_pairs = [(fp, int(m[-1])) for fp, m in matches if len(m) > 0]
    if len(file_itr_pairs) == 0:
        return None, None
    return max(file_itr_pairs, key=lambda t: t[1])


def dict_from_module(module):
    return {k: getattr(module, k) for k in module.__all__}
