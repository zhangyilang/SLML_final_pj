import os
import numpy as np


def get_feature_batch(index, dir):
    """
    A function used to read data
    :param index: a list storing the index of a batch
    :param dir: str marking the directory
    :return: a list containing a batch
    """
    names = [os.path.join(dir, str(int(i)) + '.npy') for i in index]
    batch = np.zeros([len(index), 1208, 4096], dtype=np.float32)
    for n in range(len(index)):
        f = np.load(names[n])
        batch[n, 0:f.shape[0], :] = f
    return batch


def format_time(time):
    """
    Formats a datetime to print it
    :param time: datetime
    :return: a formatted string representing time
    """
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))
