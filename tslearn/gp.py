"""
The :mod:`tslearn.svm` module contains 
Gaussian Process Regressor (GPR) model for time series.
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.gaussian_process.kernels import RBF
import numpy

from tslearn.metrics import VARIABLE_LENGTH_METRICS
from tslearn.utils import to_time_series_dataset, check_dims
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

import warnings

__author__ = 'Chester Holtz chholtz[at]eng.ucsd.edu'


def _prepare_ts_datasets_sklearn(X):
    """Prepare time series datasets for sklearn.

    Examples
    --------
    >>> X = to_time_series_dataset([[1, 2, 3], [2, 2, 3]])
    >>> _prepare_ts_datasets_sklearn(X).shape
    (2, 3)
    """
    sklearn_X = to_time_series_dataset(X)
    n_ts, sz, d = sklearn_X.shape
    return sklearn_X.reshape((n_ts, -1))


class TimeSeriesGPRMixin:
    def _preprocess_sklearn(self, X, y=None, fit_time=False):
        force_all_finite = self.kernel not in VARIABLE_LENGTH_METRICS
        if y is None:
            X = check_array(X, allow_nd=True,
                            force_all_finite=force_all_finite)
        else:
            X, y = check_X_y(X, y, allow_nd=True,
                             force_all_finite=force_all_finite)
        X = check_dims(X, X_fit=None)
        X = to_time_series_dataset(X)

        if fit_time:
            self._X_fit = X
            self.classes_ = numpy.unique(y)

        sklearn_X = _prepare_ts_datasets_sklearn(X)

        if y is None:
            return sklearn_X
        else:
            return sklearn_X, y

class TimeSeriesGPR(TimeSeriesGPRMixin, BaseEstimator, RegressorMixin):
    """Time-series specific Gaussian Process Regressor.

    Parameters
    ----------
    kernel : kernel object, optional (default=1.0 * RBF(1.0))
         Specifies the kernel type to be used in the algorithm.
         It must be one a kernel accepted by 
         ``sklearn.gaussian_process.GaussianProcessRegressor``.
         If none is given, 1.0 * RBF(1.0)) will be used. If a callable is given it is
         used to pre-compute the kernel matrix from data matrices via 
         ``sklearn.gaussian_process.kernels.PairwiseKernel``.

    alpha : int, optional (default=1e-10)
        Constant added to the diagonal of the kernel. Larger values correspond
        to additional noise-regularization. Also ensure numerical kernel is PSD.

    Attributes
    ----------
    kernel_ : kernel object
        kernel with optimized hyperparameters

    alpha_ : array, shape = [1, n_samples]
        Dual coefficients of training data points in kernel space

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of self.kernel_.theta

    gp_estimator_ : sklearn.gaussian_process.GaussianProcessRegressor
        The underlying sklearn estimator

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=64, d=2, n_blobs=2)
    >>> import numpy
    >>> y = y.astype(numpy.float) + numpy.random.randn(20) * .1
    >>> reg = TimeSeriesGPR(kernel="RBF")
    >>> reg.fit(X, y).predict(X).shape
    (20,)
    """
    def __init__(self, kernel=1.0 * RBF(1.0), alpha=1e-10, 
                 optimizer="fmin_l_bfgs_b", n_restarts=0, 
                 normalize_y=False, copy_X_train=False,
                 random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X, y):
        sklearn_X, y = self._preprocess_sklearn(X, y, fit_time=True)

        self.gp_estimator_ = GaussianProcessRegressor(
            kernel=self.kernel, alpha=self.alpha,
            optimizer=self.optimizer, n_restarts_optimizer=self.n_restarts,
            normalize_y=self.normalize_y, copy_X_train=self.copy_X_train,
            random_state=self.random_state
        )
        self.gp_estimator_.fit(sklearn_X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['gp_estimator_', '_X_fit'])
        sklearn_X = self._preprocess_sklearn(X, fit_time=False)
        return self.gp_estimator_.predict(sklearn_X)

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self, ['gp_estimator_', '_X_fit'])
        sklearn_X = self._preprocess_sklearn(X, fit_time=False)
        return self.gp_estimator_.score(sklearn_X, y,
                                         sample_weight=sample_weight)

    def _get_tags(self):
        return {'non_deterministic': True, 'allow_nan': True,
                'allow_variable_length': True}
