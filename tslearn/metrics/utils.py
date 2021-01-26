import numpy
from joblib import Parallel, delayed, parallel_backend
from tslearn.utils import to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _cdist_generic(dist_fun, dataset1, dataset2, n_jobs, verbose,
                   compute_diagonal=True, dtype=numpy.float, *args, **kwargs):
    """Compute cross-similarity matrix with joblib parallelization for a given
    similarity function.
    Parameters
    ----------
    dist_fun : function
        Similarity function to be used
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.
    compute_diagonal : bool (default: True)
        Whether diagonal terms should be computed or assumed to be 0 in the
        self-similarity case. Used only if `dataset2` is `None`.
    *args and **kwargs :
        Optional additional parameters to be passed to the similarity function.
    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix
    """  # noqa: E501
    #dataset1 = to_time_series_dataset(dataset1, dtype=dtype)

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = numpy.zeros((len(dataset1), len(dataset1)))
        indices = numpy.triu_indices(len(dataset1),
                                     k=0 if compute_diagonal else 1,
                                     m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs,
                                   prefer="processes",
                                   verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset1[j],
                *args, **kwargs
            )
            for i in range(len(dataset1))
            for j in range(i if compute_diagonal else i + 1,
                           len(dataset1))
        )
        indices = numpy.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        matrix[indices] = matrix.T[indices]
        return matrix
    else:
        #dataset2 = to_time_series_dataset(dataset2, dtype=dtype)
        with parallel_backend('loky'):
            matrix = Parallel(backend='loky', n_jobs=n_jobs, prefer="processes", verbose=verbose)(
                delayed(dist_fun)(
                    dataset1[i], dataset2[j],
                    *args, **kwargs
                )
                for i in range(len(dataset1)) for j in range(len(dataset2))
            )
        return numpy.array(matrix).reshape((len(dataset1), -1))
