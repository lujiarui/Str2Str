"""
Utilities for multiprocessing.
"""

import multiprocessing as mp

import numpy as np


def mp_map(func, args, n_cpu=None, verbose=True):
    """
    Multiprocessing wrapper for map function.
    """
    if n_cpu is None:
        n_cpu = mp.cpu_count()
    pool = mp.Pool(n_cpu)
    if verbose:
        print('Running on {} CPUs'.format(n_cpu))
    result = pool.map(func, args)
    pool.close()
    pool.join()
    return result

def parse_mp_result(result_list, sort_fn=None):
    """Parse output from mp_map"""
    if sort_fn is None:
        return result_list
    result = sorted(result_list, key=sort_fn, ) # ascending
    return result


# unit test
if __name__ == '__main__':
    pass