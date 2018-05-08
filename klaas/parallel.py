from multiprocessing import Pool, cpu_count
import numpy as np


def parallelize_array_computation(func, *arrays, n_jobs=-1):
    '''
    Chunk arrays into n_jobs blocks and compute func using a multiprocessing.Pool
    '''
    if n_jobs == -1:
        n_jobs = cpu_count()

    if n_jobs == 1:
        return func(*arrays)

    n_elements = list(set(len(a) for a in arrays))
    if len(n_elements) > 1:
        raise ValueError('All arays must have same length')
    n_elements = n_elements[0]

    blocks = []
    block_size = int(np.ceil(n_elements / n_jobs))
    for start in range(0, n_elements, block_size):
        end = start + block_size
        blocks.append([a[start:end] for a in arrays])

    with Pool(n_jobs) as pool:
        result = pool.starmap(func, blocks)

    return result
