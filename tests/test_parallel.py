import numpy as np
from multiprocessing import cpu_count


def test_correct_length():
    from aict_tools.parallel import parallelize_array_computation

    n_jobs = cpu_count()
    N = 100 * n_jobs + 1

    assert N % n_jobs != 0

    data = np.arange(N)
    results = parallelize_array_computation(np.sqrt, data, n_jobs=n_jobs)

    assert len(results) == n_jobs
    assert len(np.concatenate(results)) == N
