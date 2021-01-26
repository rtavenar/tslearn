import warnings

import numpy
from numba import njit, prange
from sklearn.metrics.pairwise import pairwise_distances

from tslearn.utils import to_time_series
from .utils import _cdist_generic

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}


@njit()
def _local_squared_dist(x, y):
    dist = 0.
    for di in range(x.shape[0]):
        diff = (x[di] - y[di])
        dist += diff * diff
    return dist


@njit()
def njit_accumulated_matrix(s1, s2, mask):
    """Compute the accumulated cost matrix score between two time series.
    
    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.

    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            if numpy.isfinite(mask[i, j]):
                cum_sum[i + 1, j + 1] = _local_squared_dist(s1[i], s2[j])
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                             cum_sum[i + 1, j],
                                             cum_sum[i, j])
    return cum_sum[1:, 1:]


@njit(nogil=True)
def njit_dtw(s1, s2, mask):
    """Compute the dynamic time warping score between two time series.

    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    dtw_score : float
        Dynamic Time Warping score between both time series.

    """
    cum_sum = njit_accumulated_matrix(s1, s2, mask)
    return numpy.sqrt(cum_sum[-1, -1])


@njit()
def _return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def dtw_path(s1, s2, global_constraint=None, sakoe_chiba_radius=None,
             itakura_max_slope=None):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return both the path and the
    similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} (X_{i} - Y_{j})^2}

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2

    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> dist
    0.0
    >>> dtw_path([1, 2, 3], [1., 2., 2., 3., 4.])[1]
    1.0

    See Also
    --------
    dtw : Get only the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets
    dtw_path_from_metric : Compute a DTW using a user-defined distance metric

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    mask = compute_mask(
        s1, s2, GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius, itakura_max_slope
    )
    acc_cost_mat = njit_accumulated_matrix(s1, s2, mask=mask)
    path = _return_path(acc_cost_mat)
    return path, numpy.sqrt(acc_cost_mat[-1, -1])


@njit()
def njit_accumulated_matrix_from_dist_matrix(dist_matrix, mask):
    """Compute the accumulated cost matrix score between two time series using
    a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : array, shape = (sz1, sz2)
        Array containing the pairwise distances.

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.

    """
    l1, l2 = dist_matrix.shape
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.

    for i in prange(l1):
        for j in prange(l2):
            if numpy.isfinite(mask[i, j]):
                cum_sum[i + 1, j + 1] = dist_matrix[i, j]
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                             cum_sum[i + 1, j],
                                             cum_sum[i, j])
    return cum_sum[1:, 1:]


def dtw_path_from_metric(s1, s2=None, metric="euclidean",
                         global_constraint=None, sakoe_chiba_radius=None,
                         itakura_max_slope=None, **kwds):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series using a distance metric defined by
    the user and return both the path and the similarity.

    Similarity is computed as the cumulative cost along the aligned time
    series.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.

    Valid values for metric are the same as for scikit-learn
    `pairwise_distances`_ function i.e. a string (e.g. "euclidean",
    "sqeuclidean", "hamming") or a function that is used to compute the
    pairwise distances. See `scikit`_ and `scipy`_ documentations for more
    information about the available metrics.

    Parameters
    ----------
    s1 : array, shape = (sz1, d) if metric!="precomputed", (sz1, sz2) otherwise
        A time series or an array of pairwise distances between samples.

    s2 : array, shape = (sz2, d), optional (default: None)
        A second time series, only allowed if metric != "precomputed".

    metric : string or callable (default: "euclidean")
        Function used to compute the pairwise distances between each points of
        `s1` and `s2`.

        If metric is "precomputed", `s1` is assumed to be a distance matrix.

        If metric is an other string, it must be one of the options compatible
        with sklearn.metrics.pairwise_distances.

        Alternatively, if metric is a callable function, it is called on pairs
        of rows of `s1` and `s2`. The callable should take two 1 dimensional
        arrays as input and return a value indicating the distance between
        them.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    **kwds
        Additional arguments to pass to sklearn pairwise_distances to compute
        the pairwise distances.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score (sum of metric along the wrapped time series).

    Examples
    --------
    Lets create 2 numpy arrays to wrap:

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> s1, s2 = rng.rand(5, 2), rng.rand(6, 2)

    The wrapping can be done by passing a string indicating the metric to pass
    to scikit-learn pairwise_distances:

    >>> dtw_path_from_metric(s1, s2,
    ...                      metric="sqeuclidean")  # doctest: +ELLIPSIS
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Or by defining a custom distance function:

    >>> sqeuclidean = lambda x, y: np.sum((x-y)**2)
    >>> dtw_path_from_metric(s1, s2, metric=sqeuclidean)  # doctest: +ELLIPSIS
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Or by using a precomputed distance matrix as input:

    >>> from sklearn.metrics.pairwise import pairwise_distances
    >>> dist_matrix = pairwise_distances(s1, s2, metric="sqeuclidean")
    >>> dtw_path_from_metric(dist_matrix,
    ...                      metric="precomputed")  # doctest: +ELLIPSIS
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Notes
    --------
    By using a squared euclidean distance metric as shown above, the output
    path is the same as the one obtained by using dtw_path but the similarity
    score is the sum of squared distances instead of the euclidean distance.

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    .. _pairwise_distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    .. _scikit: https://scikit-learn.org/stable/modules/metrics.html

    .. _scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """  # noqa: E501
    if metric == "precomputed":  # Pairwise distance given as input
        sz1, sz2 = s1.shape
        mask = compute_mask(
            sz1, sz2, GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius, itakura_max_slope
        )
        dist_mat = s1
    else:
        s1 = to_time_series(s1, remove_nans=True)
        s2 = to_time_series(s2, remove_nans=True)
        mask = compute_mask(
            s1, s2, GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius, itakura_max_slope
        )
        dist_mat = pairwise_distances(s1, s2, metric=metric, **kwds)

    acc_cost_mat = njit_accumulated_matrix_from_dist_matrix(dist_mat, mask)
    path = _return_path(acc_cost_mat)
    return path, acc_cost_mat[-1, -1]


def dtw(s1, s2, global_constraint=None, sakoe_chiba_radius=None,
        itakura_max_slope=None):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return it.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> dtw([1, 2, 3], [1., 2., 2., 3.])
    0.0
    >>> dtw([1, 2, 3], [1., 2., 2., 3., 4.])
    1.0

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    mask = compute_mask(
        s1, s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope)
    return njit_dtw(s1, s2, mask=mask)


def _max_steps(i, j, max_length, length_1, length_2):
    """Maximum number of steps required in a L-DTW process to reach a given
    cell.

    Parameters
    ----------
    i : int
        Cell row index

    j : int
        Cell column index

    max_length : int
        Maximum allowed length

    length_1 : int
        Length of the first time series

    length_2 : int
        Length of the second time series

    Returns
    -------
    int
        Number of steps
    """
    candidate_1 = i + j
    candidate_2 = max_length - max(length_1 - i - 1, length_2 - j - 1)
    return min(candidate_1, candidate_2)


def _limited_warping_length_cost(s1, s2, max_length):
    r"""Compute accumulated scores necessary fo L-DTW.

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    max_length : int
        Maximum allowed warping path length. Should be an integer between
        XXX and YYY.  # TODO

    Returns
    -------
    dict
        Accumulated scores. This dict associates (i, j) pairs (keys) to
        dictionnaries with desired length as key and associated score as value.
    """
    dict_costs = {}
    for i in range(s1.shape[0]):
        for j in range(s2.shape[0]):
            dict_costs[i, j] = {}

    # Init
    dict_costs[0, 0][0] = _local_squared_dist(s1[0], s2[0])
    for i in range(1, s1.shape[0]):
        pred = dict_costs[i - 1, 0][i - 1]
        dict_costs[i, 0][i] = pred + _local_squared_dist(s1[i], s2[0])
    for j in range(1, s2.shape[0]):
        pred = dict_costs[0, j - 1][j - 1]
        dict_costs[0, j][j] = pred + _local_squared_dist(s1[0], s2[j])

    # Main loop
    for i in range(1, s1.shape[0]):
        for j in range(1, s2.shape[0]):
            min_s = max(i, j)
            max_s = _max_steps(i, j, max_length - 1, s1.shape[0], s2.shape[0])
            for s in range(min_s, max_s + 1):
                dict_costs[i, j][s] = _local_squared_dist(s1[i], s2[j])
                dict_costs[i, j][s] += min(
                    dict_costs[i, j - 1].get(s - 1, numpy.inf),
                    dict_costs[i - 1, j].get(s - 1, numpy.inf),
                    dict_costs[i - 1, j - 1].get(s - 1, numpy.inf)
                )
    return dict_costs


def dtw_limited_warping_length(s1, s2, max_length):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series under an upper bound constraint on
    the resulting path length and return the similarity cost.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.
    This constrained-length variant was introduced in [2]_.
    Both bariants are
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    max_length : int
        Maximum allowed warping path length.
        If greater than len(s1) + len(s2), then it is equivalent to
        unconstrained DTW.
        If lower than max(len(s1), len(s2)), no path can be found and a
        ValueError is raised.

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> dtw_limited_warping_length([1, 2, 3], [1., 2., 2., 3.], 5)
    0.0
    >>> dtw_limited_warping_length([1, 2, 3], [1., 2., 2., 3., 4.], 5)
    1.0

    See Also
    --------
    dtw : Get the similarity score for DTW
    dtw_path_limited_warping_length : Get both the warping path and the
        similarity score for DTW with limited warping path length

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    .. [2] Z. Zhang, R. Tavenard, A. Bailly, X. Tang, P. Tang, T. Corpetti
           Dynamic time warping under limited warping path length.
           Information Sciences, vol. 393, pp. 91--107, 2017.
    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if max_length < max(s1.shape[0], s2.shape[0]):
        raise ValueError("Cannot find a path of length {} to align given "
                         "time series.".format(max_length))

    accumulated_costs = _limited_warping_length_cost(s1, s2, max_length)
    idx_pair = (s1.shape[0] - 1, s2.shape[0] - 1)
    optimal_cost = min(accumulated_costs[idx_pair].values())
    return numpy.sqrt(optimal_cost)


def _return_path_limited_warping_length(accum_costs, target_indices,
                                        optimal_length):
    path = [target_indices]
    cur_length = optimal_length
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array(
                [accum_costs[i - 1, j - 1].get(cur_length - 1, numpy.inf),
                 accum_costs[i - 1, j].get(cur_length - 1, numpy.inf),
                 accum_costs[i, j - 1].get(cur_length - 1, numpy.inf)]
            )
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
            cur_length -= 1
    return path[::-1]


def dtw_path_limited_warping_length(s1, s2, max_length):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series under an upper bound constraint on
    the resulting path length and return the path as well as the similarity
    cost.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.
    This constrained-length variant was introduced in [2]_.
    Both variants are
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    max_length : int
        Maximum allowed warping path length.
        If greater than len(s1) + len(s2), then it is equivalent to
        unconstrained DTW.
        If lower than max(len(s1), len(s2)), no path can be found and a
        ValueError is raised.

    Returns
    -------
    list of integer pairs
        Optimal path

    float
        Similarity score

    Examples
    --------
    >>> path, cost = dtw_path_limited_warping_length([1, 2, 3],
    ...                                              [1., 2., 2., 3.], 5)
    >>> cost
    0.0
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> path, cost = dtw_path_limited_warping_length([1, 2, 3],
    ...                                              [1., 2., 2., 3., 4.], 5)
    >>> cost
    1.0
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4)]

    See Also
    --------
    dtw_limited_warping_length : Get the similarity score for DTW with limited
        warping path length
    dtw_path : Get both the matching path and the similarity score for DTW

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    .. [2] Z. Zhang, R. Tavenard, A. Bailly, X. Tang, P. Tang, T. Corpetti
           Dynamic time warping under limited warping path length.
           Information Sciences, vol. 393, pp. 91--107, 2017.
    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if max_length < max(s1.shape[0], s2.shape[0]):
        raise ValueError("Cannot find a path of length {} to align given "
                         "time series.".format(max_length))

    accumulated_costs = _limited_warping_length_cost(s1, s2, max_length)
    idx_pair = (s1.shape[0] - 1, s2.shape[0] - 1)
    optimal_length = -1
    optimal_cost = numpy.inf
    for k, v in accumulated_costs[idx_pair].items():
        if v < optimal_cost:
            optimal_cost = v
            optimal_length = k
    path = _return_path_limited_warping_length(accumulated_costs,
                                               idx_pair,
                                               optimal_length)
    return path, numpy.sqrt(optimal_cost)


@njit()
def _subsequence_cost_matrix(subseq, longseq):
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, :] = 0.

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _local_squared_dist(subseq[i], longseq[j])
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                         cum_sum[i + 1, j],
                                         cum_sum[i, j])
    return cum_sum[1:, 1:]


def subsequence_cost_matrix(subseq, longseq):
    """Compute the accumulated cost matrix score between a subsequence and
    a reference time series.

    Parameters
    ----------
    subseq : array, shape = (sz1, d)
        Subsequence time series.

    longseq : array, shape = (sz2, d)
        Reference time series

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.
    """
    return _subsequence_cost_matrix(subseq, longseq)


@njit()
def _subsequence_path(acc_cost_mat, idx_path_end):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, idx_path_end)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def subsequence_path(acc_cost_mat, idx_path_end):
    r"""Compute the optimal path through a accumulated cost matrix given the
    endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array, shape = (sz1, sz2)
        The accumulated cost matrix comparing subsequence from a longer
        sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`

    Examples
    --------

    >>> acc_cost_mat = numpy.array([[1., 0., 0., 1., 4.],
    ...                             [5., 1., 1., 0., 1.]])
    >>> # calculate the globally optimal path
    >>> optimal_end_point = numpy.argmin(acc_cost_mat[-1, :])
    >>> path = subsequence_path(acc_cost_mat, optimal_end_point)
    >>> path
    [(0, 2), (1, 3)]

    See Also
    --------
    dtw_subsequence_path : Get the similarity score for DTW
    subsequence_cost_matrix: Calculate the required cost matrix

    """
    return _subsequence_path(acc_cost_mat, idx_path_end)


def dtw_subsequence_path(subseq, longseq):
    r"""Compute sub-sequence Dynamic Time Warping (DTW) similarity measure
    between a (possibly multidimensional) query and a long time series and
    return both the path and the similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Compared to traditional DTW, here, border constraints on admissible paths
    :math:`\pi` are relaxed such that :math:`\pi_0 = (0, ?)` and
    :math:`\pi_L = (N-1, ?)` where :math:`L` is the length of the considered
    path and :math:`N` is the length of the subsequence time series.

    It is not required that both time series share the same size, but they must
    be the same dimension. This implementation finds the best matching starting
    and ending positions for `subseq` inside `longseq`.

    Parameters
    ----------
    subseq : array, shape = (sz1, d)
        A query time series.
    longseq : array, shape = (sz2, d)
        A reference (supposed to be longer than `subseq`) time series.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`.
    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_subsequence_path([2., 3.], [1., 2., 2., 3., 4.])
    >>> path
    [(0, 2), (1, 3)]
    >>> dist
    0.0

    See Also
    --------
    dtw : Get the similarity score for DTW
    subsequence_cost_matrix: Calculate the required cost matrix
    subsequence_path: Calculate a matching path manually
    """
    subseq = to_time_series(subseq)
    longseq = to_time_series(longseq)
    acc_cost_mat = subsequence_cost_matrix(subseq=subseq,
                                           longseq=longseq)
    global_optimal_match = numpy.argmin(acc_cost_mat[-1, :])
    path = subsequence_path(acc_cost_mat, global_optimal_match)
    return path, numpy.sqrt(acc_cost_mat[-1, :][global_optimal_match])


@njit()
def sakoe_chiba_mask(sz1, sz2, radius=1):
    """Compute the Sakoe-Chiba mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    radius : int
        The radius of the band.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Sakoe-Chiba mask.

    Examples
    --------
    >>> sakoe_chiba_mask(4, 4, 1)
    array([[ 0.,  0., inf, inf],
           [ 0.,  0.,  0., inf],
           [inf,  0.,  0.,  0.],
           [inf, inf,  0.,  0.]])
    >>> sakoe_chiba_mask(7, 3, 1)
    array([[ 0.,  0., inf],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [inf,  0.,  0.]])
    """
    mask = numpy.full((sz1, sz2), numpy.inf)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in prange(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = 0.
    else:
        width = sz2 - sz1 + radius
        for i in prange(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = 0.
    return mask


@njit()
def _njit_itakura_mask(sz1, sz2, max_slope=2.):
    """Compute the Itakura mask without checking that the constraints
    are feasible. In most cases, you should use itakura_mask instead.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    max_slope : float (default = 2)
        The maximum slope of the parallelogram.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Itakura mask.
    """
    min_slope = 1 / float(max_slope)
    max_slope *= (float(sz1) / float(sz2))
    min_slope *= (float(sz1) / float(sz2))

    lower_bound = numpy.empty((2, sz2))
    lower_bound[0] = min_slope * numpy.arange(sz2)
    lower_bound[1] = ((sz1 - 1) - max_slope * (sz2 - 1)
                      + max_slope * numpy.arange(sz2))
    lower_bound_ = numpy.empty(sz2)
    for i in prange(sz2):
        lower_bound_[i] = max(round(lower_bound[0, i], 2),
                              round(lower_bound[1, i], 2))
    lower_bound_ = numpy.ceil(lower_bound_)

    upper_bound = numpy.empty((2, sz2))
    upper_bound[0] = max_slope * numpy.arange(sz2)
    upper_bound[1] = ((sz1 - 1) - min_slope * (sz2 - 1)
                      + min_slope * numpy.arange(sz2))
    upper_bound_ = numpy.empty(sz2)
    for i in prange(sz2):
        upper_bound_[i] = min(round(upper_bound[0, i], 2),
                              round(upper_bound[1, i], 2))
    upper_bound_ = numpy.floor(upper_bound_ + 1)

    mask = numpy.full((sz1, sz2), numpy.inf)
    for i in prange(sz2):
        mask[int(lower_bound_[i]):int(upper_bound_[i]), i] = 0.
    return mask


def itakura_mask(sz1, sz2, max_slope=2.):
    """Compute the Itakura mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    max_slope : float (default = 2)
        The maximum slope of the parallelogram.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Itakura mask.

    Examples
    --------
    >>> itakura_mask(6, 6)
    array([[ 0., inf, inf, inf, inf, inf],
           [inf,  0.,  0., inf, inf, inf],
           [inf,  0.,  0.,  0., inf, inf],
           [inf, inf,  0.,  0.,  0., inf],
           [inf, inf, inf,  0.,  0., inf],
           [inf, inf, inf, inf, inf,  0.]])
    """
    mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)

    # Post-check
    raise_warning = False
    for i in prange(sz1):
        if not numpy.any(numpy.isfinite(mask[i])):
            raise_warning = True
            break
    if not raise_warning:
        for j in prange(sz2):
            if not numpy.any(numpy.isfinite(mask[:, j])):
                raise_warning = True
                break
    if raise_warning:
        warnings.warn("'itakura_max_slope' constraint is unfeasible "
                      "(ie. leads to no admissible path) for the "
                      "provided time series sizes",
                      RuntimeWarning)

    return mask


def compute_mask(s1, s2, global_constraint=0,
                 sakoe_chiba_radius=None, itakura_max_slope=None):
    """Compute the mask (region constraint).

    Parameters
    ----------
    s1 : array
        A time series or integer.

    s2: array
        Another time series or integer.

    global_constraint : {0, 1, 2} (default: 0)
        Global constraint to restrict admissible paths for DTW:
        - "itakura" if 1
        - "sakoe_chiba" if 2
        - no constraint otherwise

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to 2 (sakoe-chiba), a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to 1 (itakura), a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    Returns
    -------
    mask : array
        Constraint region.
    """
    # The output mask will be of shape (sz1, sz2)
    if isinstance(s1, int) and isinstance(s2, int):
        sz1, sz2 = s1, s2
    else:
        sz1 = s1.shape[0]
        sz2 = s2.shape[0]
    if (global_constraint == 0 and sakoe_chiba_radius is not None
            and itakura_max_slope is not None):
        raise RuntimeWarning("global_constraint is not set for DTW, but both "
                             "sakoe_chiba_radius and itakura_max_slope are "
                             "set, hence global_constraint cannot be inferred "
                             "and no global constraint will be used.")
    if global_constraint == 2 or (global_constraint == 0
                                  and sakoe_chiba_radius is not None):
        if sakoe_chiba_radius is None:
            sakoe_chiba_radius = 1
        mask = sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius)
    elif global_constraint == 1 or (global_constraint == 0
                                    and itakura_max_slope is not None):
        if itakura_max_slope is None:
            itakura_max_slope = 2.
        mask = itakura_mask(sz1, sz2, max_slope=itakura_max_slope)
    else:
        mask = numpy.zeros((sz1, sz2))
    return mask


def cdist_dtw(dataset1, dataset2=None, global_constraint=None,
              sakoe_chiba_radius=None, itakura_max_slope=None, n_jobs=None,
              verbose=0):
    r"""Compute cross-similarity matrix using Dynamic Time Warping (DTW)
    similarity measure.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} (X_{i} - Y_{j})^2}

    DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series

    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

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

    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]])
    array([[0., 1.],
           [1., 0.]])
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 3], [2, 3, 4, 5]])
    array([[0.        , 2.44948974],
           [1.        , 1.41421356]])

    See Also
    --------
    dtw : Get DTW similarity score

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """  # noqa: E501
    return _cdist_generic(dist_fun=dtw, dataset1=dataset1, dataset2=dataset2,
                          n_jobs=n_jobs, verbose=verbose,
                          compute_diagonal=False,
                          global_constraint=global_constraint,
                          sakoe_chiba_radius=sakoe_chiba_radius,
                          itakura_max_slope=itakura_max_slope)

def lb_keogh(ts_query, ts_candidate=None, radius=1, envelope_candidate=None):
    r"""Compute LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts_query : array-like
        Query time-series to compare to the envelope of the candidate.
    ts_candidate : array-like or None (default: None)
        Candidate time-series. None means the envelope is provided via
        `envelope_candidate` parameter and hence does not
        need to be computed again.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on
        all observations from the candidate time series at indices comprised
        between i-radius and i+radius). Not used
        if `ts_candidate` is None.
    envelope_candidate: pair of array-like (envelope_down, envelope_up) or None
    (default: None)
        Pre-computed envelope of the candidate time series. If set to None, it
        is computed based on `ts_candidate`.

    Notes
    -----
        This method requires a `ts_query` and `ts_candidate` (or
        `envelope_candidate`, depending on the call) to be of equal size.

    Returns
    -------
    float
        Distance between the query time series and the envelope of the
        candidate time series.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> ts2 = [0, 0, 0, 0, 0]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> lb_keogh(ts_query=ts2,
    ...          envelope_candidate=(env_low, env_up))  # doctest: +ELLIPSIS
    2.8284...
    >>> lb_keogh(ts_query=ts2,
    ...          ts_candidate=ts1,
    ...          radius=1)  # doctest: +ELLIPSIS
    2.8284...

    See also
    --------
    lb_envelope : Compute LB_Keogh-related envelope

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    if ts_candidate is None:
        envelope_down, envelope_up = envelope_candidate
    else:
        ts_candidate = to_time_series(ts_candidate)
        assert ts_candidate.shape[1] == 1, \
            "LB_Keogh is available only for monodimensional time series"
        envelope_down, envelope_up = lb_envelope(ts_candidate, radius)
    ts_query = to_time_series(ts_query)
    assert ts_query.shape[1] == 1, \
        "LB_Keogh is available only for monodimensional time series"
    indices_up = ts_query[:, 0] > envelope_up[:, 0]
    indices_down = ts_query[:, 0] < envelope_down[:, 0]
    return numpy.sqrt(numpy.linalg.norm(ts_query[indices_up, 0] -
                                        envelope_up[indices_up, 0]) ** 2 +
                      numpy.linalg.norm(ts_query[indices_down, 0] -
                                        envelope_down[indices_down, 0]) ** 2)


@njit()
def njit_lb_envelope(time_series, radius):
    sz, d = time_series.shape
    enveloppe_up = numpy.empty((sz, d))
    enveloppe_down = numpy.empty((sz, d))

    for i in prange(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        for di in prange(d):
            enveloppe_down[i, di] = numpy.min(time_series[min_idx:max_idx, di])
            enveloppe_up[i, di] = numpy.max(time_series[min_idx:max_idx, di])

    return enveloppe_down, enveloppe_up


def lb_envelope(ts, radius=1):
    r"""Compute time-series envelope as required by LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts : array-like
        Time-series for which the envelope should be computed.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on
        all observations from the time series at indices comprised between
        i-radius and i+radius).

    Returns
    -------
    array-like
        Lower-side of the envelope.
    array-like
        Upper-side of the envelope.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> env_low
    array([[1.],
           [1.],
           [2.],
           [1.],
           [1.]])
    >>> env_up
    array([[2.],
           [3.],
           [3.],
           [3.],
           [2.]])

    See also
    --------
    lb_keogh : Compute LB_Keogh similarity

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    return njit_lb_envelope(to_time_series(ts), radius=radius)


@njit(nogil=True)
def njit_lcss_accumulated_matrix(s1, s2, eps, mask):
    """Compute the longest common subsequence similarity score between
    two time series.

    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series.

    eps : float
        Matching threshold.

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    lcss_score : float
        Longest Common Subsequence similarity score between both time series.

    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    acc_cost_mat = numpy.full((l1 + 1, l2 + 1), 0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if numpy.isfinite(mask[i - 1, j - 1]):
                if numpy.sqrt(_local_squared_dist(s1[i - 1], s2[j - 1])) <= eps:
                    acc_cost_mat[i][j] = 1 + acc_cost_mat[i - 1][j - 1]
                else:
                    acc_cost_mat[i][j] = max(acc_cost_mat[i][j - 1],
                                             acc_cost_mat[i - 1][j])

    return acc_cost_mat


@njit(nogil=True)
def njit_lcss(s1, s2, eps, mask):
    """Compute the longest common subsequence score between two time series.

    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series

    eps : float (default: 1.)
        Maximum matching distance threshold.

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    lcss_score : float
        Longest Common Subsquence score between both time series.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    acc_cost_mat = njit_lcss_accumulated_matrix(s1, s2, eps, mask)

    return float(acc_cost_mat[-1][-1]) / min([l1, l2])


def lcss(s1, s2, eps=1., global_constraint=None, sakoe_chiba_radius=None,
        itakura_max_slope=None):
    r"""Compute the Longest Common Subsequence (LCSS) similarity measure
    between (possibly multidimensional) time series and return the
    similarity.

    LCSS is computed by matching indexes that are met up until the eps
    threshold, so it leaves some points unmatched and focuses on the
    similar parts of two sequences. The matching can occur even if the
    time indexes are different, regulated through the delta parameter
    that defines how far it can go. To retrieve a meaningful similarity
    value from the length of the longest common subsequence, the
    percentage of that value regarding the length of the shortest time
    series is returned.

    According to this definition, the values returned by LCSS range from
    0 to 1, the highest value taken when two time series fully match,
    and vice-versa. It is not required that both time series share the
    same size, but they must be the same dimension. LCSS was originally
    presented in [1]_ and is discussed in more details in our
    :ref:`dedicated user-guide page <lcss>`.

    Note
    ----
    Contrary to Dynamic Time Warping and variants, an LCSS path does not need to be contiguous.

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    eps : float (default: 1.)
        Maximum matching distance threshold.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for LCSS.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> lcss([1, 2, 3], [1., 2., 2., 3.])
    1.0
    >>> lcss([1, 2, 3], [1., 2., 2., 4., 7.])
    1.0
    >>> lcss([1, 2, 3], [1., 2., 2., 2., 3.], eps=0)
    1.0
    >>> lcss([1, 2, 3], [-2., 5., 7.], eps=3)
    0.6666666666666666

    See Also
    --------
    lcss_path: Get both the matching path and the similarity score for LCSS

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.

    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    mask = compute_mask(
        s1, s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope)

    return njit_lcss(s1, s2, eps, mask)


@njit()
def _return_lcss_path(s1, s2, eps, mask, acc_cost_mat, sz1, sz2):
    i, j = (sz1, sz2)
    path = []

    while i > 0 and j > 0:
        if numpy.isfinite(mask[i - 1, j - 1]):
            if numpy.sqrt(_local_squared_dist(s1[i - 1], s2[j - 1])) <= eps:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif acc_cost_mat[i - 1][j] > acc_cost_mat[i][j-1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]


@njit()
def _return_lcss_path_from_dist_matrix(dist_matrix, eps, mask, acc_cost_mat, sz1, sz2):
    i, j = (sz1, sz2)
    path = []

    while i > 0 and j > 0:
        if numpy.isfinite(mask[i - 1, j - 1]):
            if dist_matrix[i - 1, j - 1] <= eps:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif acc_cost_mat[i - 1][j] > acc_cost_mat[i][j-1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]


def lcss_path(s1, s2, eps=1, global_constraint=None, sakoe_chiba_radius=None,
             itakura_max_slope=None):
    r"""Compute the Longest Common Subsequence (LCSS) similarity measure
    between (possibly multidimensional) time series and return both the
    path and the similarity.

    LCSS is computed by matching indexes that are met up until the eps
    threshold, so it leaves some points unmatched and focuses on the
    similar parts of two sequences. The matching can occur even if the
    time indexes are different, which can be regulated through the sakoe
    chiba radius parameter that defines how far it can go.

    To retrieve a meaningful similarity value from the length of the
    longest common subsequence, the percentage of that value regarding
    the length of the shortest time series is returned.

    According to this definition, the values returned by LCSS range from
    0 to 1, the highest value taken when two time series fully match,
    and vice-versa. It is not required that both time series share the
    same size, but they must be the same dimension. LCSS was originally
    presented in [1]_ and is discussed in more details in our
    :ref:`dedicated user-guide page <lcss>`.

    Note
    ----
    Contrary to Dynamic Time Warping and variants, an LCSS path does not need to be contiguous.

    Parameters
    ----------
    s1
        A time series.

    s2
        Another time series.

    eps : float (default: 1.)
        Maximum matching distance threshold.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
       Global constraint to restrict admissible paths for LCSS.

    sakoe_chiba_radius : int or None (default: None)
       Radius to be used for Sakoe-Chiba band global constraint.
       If None and `global_constraint` is set to "sakoe_chiba", a radius of
       1 is used.
       If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
       `global_constraint` is used to infer which constraint to use among the
       two. In this case, if `global_constraint` corresponds to no global
       constraint, a `RuntimeWarning` is raised and no global constraint is
       used.

    itakura_max_slope : float or None (default: None)
       Maximum slope for the Itakura parallelogram constraint.
       If None and `global_constraint` is set to "itakura", a maximum slope
       of 2. is used.
       If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
       `global_constraint` is used to infer which constraint to use among the
       two. In this case, if `global_constraint` corresponds to no global
       constraint, a `RuntimeWarning` is raised and no global constraint is
       used.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2

    float
        Similarity score

    Examples
    --------
    >>> path, sim = lcss_path([1., 2., 3.], [1., 2., 2., 3.])
    >>> path
    [(0, 1), (1, 2), (2, 3)]
    >>> sim
    1.0
    >>> lcss_path([1., 2., 3.], [1., 2., 2., 4.])[1]
    1.0

    See Also
    --------
    lcss : Get only the similarity score for LCSS
    lcss_path_from_metric: Compute LCSS using a user-defined distance metric

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.

    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    mask = compute_mask(
        s1, s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius, itakura_max_slope)

    l1 = s1.shape[0]
    l2 = s2.shape[0]

    acc_cost_mat = njit_lcss_accumulated_matrix(s1, s2, eps, mask)
    path = _return_lcss_path(s1, s2, eps, mask, acc_cost_mat, l1, l2)

    return path, float(acc_cost_mat[-1][-1]) / min([l1, l2])


def njit_lcss_accumulated_matrix_from_dist_matrix(dist_matrix, eps, mask):
    """Compute the accumulated cost matrix score between two time series using
    a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : array, shape = (sz1, sz2)
        Array containing the pairwise distances.

    eps : float (default: 1.)
        Maximum matching distance threshold.

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.

    """
    l1, l2 = dist_matrix.shape
    acc_cost_mat = numpy.full((l1 + 1, l2 + 1), 0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if numpy.isfinite(mask[i - 1, j - 1]):
                if dist_matrix[i - 1, j - 1] <= eps:
                    acc_cost_mat[i][j] = 1 + acc_cost_mat[i - 1][j - 1]
                else:
                    acc_cost_mat[i][j] = max(acc_cost_mat[i][j - 1],
                                             acc_cost_mat[i - 1][j])

    return acc_cost_mat


def lcss_path_from_metric(s1, s2=None, eps=1, metric="euclidean",
                          global_constraint=None, sakoe_chiba_radius=None,
                          itakura_max_slope=None, **kwds):
    r"""Compute the Longest Common Subsequence (LCSS) similarity measure between
    (possibly multidimensional) time series using a distance metric defined by
    the user and return both the path and the similarity.

    Having the length of the longest commom subsequence between two time-series,
    the similarity is computed as the percentage of that value regarding the
    length of the shortest time series.

    It is not required that both time series share the same size, but they must
    be the same dimension. LCSS was originally presented in [1]_.

    Valid values for metric are the same as for scikit-learn
    `pairwise_distances`_ function i.e. a string (e.g. "euclidean",
    "sqeuclidean", "hamming") or a function that is used to compute the
    pairwise distances. See `scikit`_ and `scipy`_ documentations for more
    information about the available metrics.

    Parameters
    ----------
    s1 : array, shape = (sz1, d) if metric!="precomputed", (sz1, sz2) otherwise
        A time series or an array of pairwise distances between samples.

    s2 : array, shape = (sz2, d), optional (default: None)
        A second time series, only allowed if metric != "precomputed".

    eps : float (default: 1.)
        Maximum matching distance threshold.
        
    metric : string or callable (default: "euclidean")
        Function used to compute the pairwise distances between each points of
        `s1` and `s2`.

        If metric is "precomputed", `s1` is assumed to be a distance matrix.

        If metric is an other string, it must be one of the options compatible
        with sklearn.metrics.pairwise_distances.

        Alternatively, if metric is a callable function, it is called on pairs
        of rows of `s1` and `s2`. The callable should take two 1 dimensional
        arrays as input and return a value indicating the distance between
        them.
        
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for LCSS.
        
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
        
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
        
    **kwds
        Additional arguments to pass to sklearn pairwise_distances to compute
        the pairwise distances.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score.

    Examples
    --------
    Lets create 2 numpy arrays to wrap:

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> s1, s2 = rng.rand(5, 2), rng.rand(6, 2)

    The wrapping can be done by passing a string indicating the metric to pass
    to scikit-learn pairwise_distances:

    >>> lcss_path_from_metric(s1, s2,
    ...                      metric="sqeuclidean")  # doctest: +ELLIPSIS
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.0)

    Or by defining a custom distance function:

    >>> sqeuclidean = lambda x, y: np.sum((x-y)**2)
    >>> lcss_path_from_metric(s1, s2, metric=sqeuclidean)  # doctest: +ELLIPSIS
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.0)

    Or by using a precomputed distance matrix as input:

    >>> from sklearn.metrics.pairwise import pairwise_distances
    >>> dist_matrix = pairwise_distances(s1, s2, metric="sqeuclidean")
    >>> lcss_path_from_metric(dist_matrix,
    ...                      metric="precomputed")  # doctest: +ELLIPSIS
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.0)

    Notes
    --------
    By using a squared euclidean distance metric as shown above, the output
    path and similarity is the same as the one obtained by using lcss_path 
    (which uses the euclidean distance) simply because with the sum of squared
    distances the matching threshold is still not reached.
    Also, contrary to Dynamic Time Warping and variants, an LCSS path does not need to be contiguous.
    
    See Also
    --------
    lcss: Get only the similarity score for LCSS
    lcss_path : Get both the matching path and the similarity score for LCSS

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.

    .. _pairwise_distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    .. _scikit: https://scikit-learn.org/stable/modules/metrics.html

    .. _scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """  # noqa: E501
    if metric == "precomputed":  # Pairwise distance given as input
        sz1, sz2 = s1.shape
        mask = compute_mask(
            sz1, sz2, GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius, itakura_max_slope
        )
        dist_mat = s1
    else:
        s1 = to_time_series(s1, remove_nans=True)
        s2 = to_time_series(s2, remove_nans=True)
        sz1 = s1.shape[0]
        sz2 = s2.shape[0]
        mask = compute_mask(
            s1, s2, GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius, itakura_max_slope
        )
        dist_mat = pairwise_distances(s1, s2, metric=metric, **kwds)

    acc_cost_mat = njit_lcss_accumulated_matrix_from_dist_matrix(dist_mat, eps, mask)
    path = _return_lcss_path_from_dist_matrix(dist_mat, eps, acc_cost_mat, mask, sz1, sz2)

    return path, float(acc_cost_mat[-1][-1]) / min([sz1, sz2])
