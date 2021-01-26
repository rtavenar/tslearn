import numpy
import warnings
from scipy.interpolate import interp1d
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from joblib import parallel_backend, Parallel, delayed

from ..metrics import dtw_path
from ..utils import to_time_series_dataset, ts_size
from .utils import _set_weights

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _init_avg(X, barycenter_size):
    if X.shape[1] == barycenter_size:
        return numpy.nanmean(X, axis=0)
    else:
        X_avg = numpy.nanmean(X, axis=0)
        xnew = numpy.linspace(0, 1, barycenter_size)
        f = interp1d(numpy.linspace(0, 1, X_avg.shape[0]), X_avg,
                     kind="linear", axis=0)
        return f(xnew)


def _petitjean_assignment(X, barycenter, metric_params=None):
    if metric_params is None:
        metric_params = {}
    n = X.shape[0]
    barycenter_size = barycenter.shape[0]
    assign = ([[] for _ in range(barycenter_size)],
              [[] for _ in range(barycenter_size)])
    for i in range(n):
        path, _ = dtw_path(X[i], barycenter, **metric_params)
        for pair in path:
            assign[0][pair[1]].append(i)
            assign[1][pair[1]].append(pair[0])
    return assign


def _petitjean_update_barycenter(X, assign, barycenter_size, weights):
    barycenter = numpy.zeros((barycenter_size, X.shape[-1]))
    for t in range(barycenter_size):
        barycenter[t] = numpy.average(X[assign[0][t], assign[1][t]], axis=0,
                                      weights=weights[assign[0][t]])
    return barycenter


def _petitjean_cost(X, barycenter, assign, weights):
    cost = 0.
    barycenter_size = barycenter.shape[0]
    for t_barycenter in range(barycenter_size):
        for i_ts, t_ts in zip(assign[0][t_barycenter],
                              assign[1][t_barycenter]):
            sq_norm = numpy.linalg.norm(X[i_ts, t_ts] -
                                        barycenter[t_barycenter]) ** 2
            cost += weights[i_ts] * sq_norm
    return cost / weights.sum()


def dtw_barycenter_averaging_petitjean(X, barycenter_size=None,
                                       init_barycenter=None,
                                       max_iter=30, tol=1e-5, weights=None,
                                       metric_params=None,
                                       verbose=False):
    """DTW Barycenter Averaging (DBA) method.

    DBA was originally presented in [1]_.
    This implementation is not the one documented in the API, but is kept
    in the codebase to check the documented one for non-regression.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3. ],
           [4. ],
           [4.5]])
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5,
    ...                          metric_params={"itakura_max_slope": 2})
    array([[1. ],
           [2. ],
           [3. ],
           [3.5],
           [4.5]])
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5,
    ...                                    barycenter_size=3)
    array([[1.5       ],
           [3.        ],
           [4.33333333]])
    >>> dtw_barycenter_averaging_petitjean([[0, 0, 0], [10, 10, 10]],
    ...                                    weights=numpy.array([0.75, 0.25]))
    array([[2.5],
           [2.5],
           [2.5]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    #X_ = to_time_series_dataset(X)
    X_ = X
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    for it in range(max_iter):
        assign = _petitjean_assignment(X_, barycenter, metric_params)
        cost = _petitjean_cost(X_, barycenter, assign, weights)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        barycenter = _petitjean_update_barycenter(X_, assign, barycenter_size,
                                                  weights)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter


def _mm_assignment(X, barycenter, weights, metric_params=None, n_treads=15):
    """Computes item assignement based on DTW alignments and return cost as a
    bonus.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the current step of the algorithm.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    Returns
    -------
    list of index pairs
        Warping paths

    float
        Current alignment cost
    """
    if metric_params is None:
        metric_params = {}
    n = X.shape[0]
    cost = 0.
    list_p_k = []
    #with parallel_backend('threading'):
    res = Parallel(backend='threading',prefer="threads",require='sharedmem',n_jobs=n_treads,verbose=10)(delayed(dtw_path)(barycenter, X[i], global_constraint="sakoe_chiba",sakoe_chiba_radius=1) for i in range(n))
    paths, dists = zip(*res)
    paths = list(paths)
    dists = list(dists)
    cost = numpy.sum(numpy.multiply(numpy.power(dists, 2),weights))
    cost /= weights.sum()
    list_p_k = paths
    #for i in range(n):
    #    path, dist_i = dtw_path(barycenter, X[i], **metric_params)
    #    cost += dist_i ** 2 * weights[i]
    #    list_p_k.append(path)
    #cost /= weights.sum()
    return list_p_k, cost


def _subgradient_valence_warping(list_p_k, barycenter_size, weights, n_treads):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns a list of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_p_k : list of index pairs
        Warping paths

    barycenter_size : int
        Size of the barycenter to generate.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    list of numpy.array of shape (barycenter_size, )
        list of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    def create_w_k(p_k, barycenter_size):
        w_k = numpy.zeros((barycenter_size, p_k[-1][1] + 1))
        for i, j in p_k:
            w_k[i, j] = 1.
        return w_k
    w_k_ones = Parallel(backend='threading',prefer='threads',require='sharedmem', n_jobs=n_treads, verbose = True)(delayed(create_w_k)(p_k,barycenter_size) for p_k in list_p_k)
    def mult_w_k_weights(w_k, weight):
        return w_k * weight
    list_w_k = Parallel(backend='threading',prefer='threads',require='sharedmem', n_jobs=n_treads, verbose = True)(delayed(mult_w_k_weights)(w_k,weights[k]) for k,w_k in enumerate(w_k_ones))
    def mult_w_k_sum_wieghts(w_k, weight):
        return w_k.sum(axis=1) * weight
    list_v_k = Parallel(backend='threading',prefer='threads',require='sharedmem', n_jobs=n_treads, verbose = True)(delayed(mult_w_k_sum_wieghts)(w_k,weight) for (w_k, weight) in zip(w_k_ones, weights))
    return list_v_k, list_w_k


def _mm_valence_warping(list_p_k, barycenter_size, weights, n_treads):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns the sum of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_p_k : list of index pairs
        Warping paths

    barycenter_size : int
        Size of the barycenter to generate.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    list_v_k, list_w_k = _subgradient_valence_warping(
        list_p_k=list_p_k,
        barycenter_size=barycenter_size,
        weights=weights)
    #diag_sum_v_k = numpy.zeros(list_v_k[0].shape)
    #for v_k in list_v_k:
    #    diag_sum_v_k += v_k
    diag_sum_v_k = Parallel(backend='threading',prefer='threads',require='sharedmem', n_jobs=n_treads, verbose = True)(delayed(numpy.sum)(x) for x in zip(*numpy.array(list_v_k)))
    diag_sum_v_k = numpy.array(diag_sum_v_k)
    return diag_sum_v_k, list_w_k


def _mm_update_barycenter(X, diag_sum_v_k, list_w_k, n_treads):
    """Update barycenters using the formula from Algorithm 2 in [1]_.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    diag_sum_v_k : numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list_w_k : list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    Returns
    -------
    numpy.array of shape (barycenter_size, d)
        Updated barycenter

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    #d = X.shape[2]
    #barycenter_size = diag_sum_v_k.shape[0]
    #sum_w_x = numpy.zeros((barycenter_size, d))
    #for k, (w_k, x_k) in enumerate(zip(list_w_k, X)):
    #    sum_w_x += w_k.dot(x_k[:ts_size(x_k)])
    sum_w_x = Parallel(backend='threading',prefer='threads',require='sharedmem', n_jobs=n_treads, verbose = True)(delayed(numpy.dot)(w_k,x_k[:ts_size(x_k)]) for (w_k, x_k) in zip(list_w_k, X))
    sum_w_x = Parallel(backend='threading',prefer='threads',require='sharedmem',n_jobs=n_treads, verbose = True)(delayed(numpy.sum)(x) for x in zip(*numpy.array(sum_w_x)))
    sum_w_x = numpy.array(sum_w_x).reshape((len(sum_w_x), 1))
    inverse_diag_sum_v_k = numpy.reciprocal(diag_sum_v_k)
    inverse_diag_sum_v_k = numpy.diag(inverse_diag_sum_v_k)
    barycenter = inverse_diag_sum_v_k.dot(sum_w_x)
    return barycenter


def _subgradient_update_barycenter(X, list_diag_v_k, list_w_k, weights_sum,
                                   barycenter, eta):
    """Update barycenters using the formula from Algorithm 1 in [1]_.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    list_diag_v_k : list of numpy.array of shape (barycenter_size, )
        list of weighted :math:`V^{(k)}` diagonals (as vectors)

    list_w_k : list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    weights_sum : float
        sum of weights applied to matrices :math:`V^{(k)}` and :math:`W^{(k)}`

    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the previous iteration of the algorithm

    eta : float
        Step-size for the subgradient descent algorithm

    Returns
    -------
    numpy.array of shape (barycenter_size, d)
        Updated barycenter

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    d = X.shape[2]
    barycenter_size = barycenter.shape[0]
    delta_bar = numpy.zeros((barycenter_size, d))
    for k, (v_k, w_k, x_k) in enumerate(zip(list_diag_v_k, list_w_k, X)):
        delta_bar += v_k.reshape((-1, 1)) * barycenter
        delta_bar -= w_k.dot(x_k[:ts_size(x_k)])
    barycenter -= (2. * eta / weights_sum) * delta_bar
    return barycenter


def dtw_barycenter_averaging(X, barycenter_size=None, init_barycenter=None,
                             max_iter=30, tol=1e-5, weights=None, n_treads=15,
                             metric_params=None,
                             verbose=False, n_init=1):
    """DTW Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    n_init : int (default: 1)
        Number of different initializations to be tried (useful only is
        init_barycenter is set to None, otherwise, all trials will reach the
        same performance)

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3. ],
           [4. ],
           [4.5]])
    >>> dtw_barycenter_averaging(time_series, max_iter=5,
    ...                          metric_params={"itakura_max_slope": 2})
    array([[1. ],
           [2. ],
           [3. ],
           [3.5],
           [4.5]])
    >>> dtw_barycenter_averaging(time_series, max_iter=5, barycenter_size=3)
    array([[1.5       ],
           [3.        ],
           [4.33333333]])
    >>> dtw_barycenter_averaging([[0, 0, 0], [10, 10, 10]], max_iter=1,
    ...                          weights=numpy.array([0.75, 0.25]))
    array([[2.5],
           [2.5],
           [2.5]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    best_cost = numpy.inf
    best_barycenter = None
    for i in range(n_init):
        if verbose:
            print("Attempt {}".format(i + 1))
        bary, loss = dtw_barycenter_averaging_one_init(
            X=X,
            barycenter_size=barycenter_size,
            init_barycenter=init_barycenter,
            max_iter=max_iter,
            tol=tol,
            weights=weights,
            metric_params=metric_params,
            verbose=verbose
        )
        if loss < best_cost:
            best_cost = loss
            best_barycenter = bary
    return best_barycenter


def dtw_barycenter_averaging_one_init(X, barycenter_size=None,
                                      init_barycenter=None,
                                      max_iter=30, tol=1e-5, weights=None,
									  n_treads=15,
                                      metric_params=None,
                                      verbose=False):
    """DTW Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.
    float
        Associated inertia

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    #X_ = to_time_series_dataset(X)
    if barycenter_size is None:
        barycenter_size = X.shape[1]
    weights = _set_weights(weights, X.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    for it in range(max_iter):
        list_p_k, cost = _mm_assignment(X, barycenter, weights, n_treads, metric_params)
        diag_sum_v_k, list_w_k = _mm_valence_warping(list_p_k, barycenter_size,
                                                     weights, n_treads)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        barycenter = _mm_update_barycenter(X, diag_sum_v_k, list_w_k, n_treads)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter, cost


def dtw_barycenter_averaging_subgradient(X, barycenter_size=None,
                                         init_barycenter=None, max_iter=30,
                                         initial_step_size=.05,
                                         final_step_size=.005,
                                         tol=1e-5, random_state=None,
                                         weights=None,
                                         metric_params=None, verbose=False):
    """DTW Barycenter Averaging (DBA) method estimated through subgradient
    descent algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Stochastic Subgradient
    Mean Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    initial_step_size : float (default: 0.05)
        Initial step size for the subgradient descent algorithm.
        Default value is the one suggested in [2]_.

    final_step_size : float (default: 0.005)
        Final step size for the subgradient descent algorithm.
        Default value is the one suggested in [2]_.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging_subgradient(
    ...     time_series,
    ...     max_iter=10,
    ...     random_state=0
    ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[1. ],
           [2. ],
           [3.5...],
           [4.5...]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    rng = check_random_state(random_state)

    #X_ = to_time_series_dataset(X)
    X_ = X
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    eta = initial_step_size
    n = X_.shape[0]
    for it in range(max_iter):
        shuffled_indices = rng.permutation(n)
        for idx in shuffled_indices:
            Xi = X_[idx:idx+1]
            wi = weights[idx:idx+1]
            list_p_k, cost = _mm_assignment(Xi, barycenter, weights,
                                            metric_params)
            list_diag_v_k, list_w_k = _subgradient_valence_warping(
                list_p_k,
                barycenter_size,
                wi
            )
            if verbose:
                print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
            barycenter = _subgradient_update_barycenter(Xi, list_diag_v_k,
                                                        list_w_k, wi.sum(),
                                                        barycenter, eta)
            if it == 0:
                eta -= (initial_step_size - final_step_size) / n
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter
