"""Functions for dealing with global variables"""
from typing import Optional
from multiprocessing import Pool

_pool: Optional[Pool] = None
"""Multiprocessing pool used across multiple processes."""
_pool_size: Optional[int] = None
"""Size of this pool"""


def get_process_pool(n_jobs: int, error_if_wrong_size: bool = False, **kwargs):
    """Access or create the current process pool

    Args:
        n_jobs: Size of the process pool
        error_if_wrong_size: Whether to throw an error if the pool exists
            and is a different size than ``n_jobs``
        kwargs: Passed to the constructor for Pool
    Returns:
        Process pool
    """
    global _pool, _pool_size

    # Make the pool if needed
    if _pool is None:
        _pool = Pool(n_jobs, **kwargs)
        return _pool

    # Return the pool
    if _pool_size != n_jobs and error_if_wrong_size:
        raise ValueError(f'Pool has been created and has {_pool_size} members, but you requested {n_jobs}')
    return _pool
