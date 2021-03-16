import numpy
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.interpolate import interp1d
import warnings

from tslearn.utils import (to_time_series_dataset, check_equal_size, ts_size,
                           check_dims)
from tslearn.bases import TimeSeriesBaseEstimator

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesResampler(TransformerMixin):
    """Resampler for time series. Resample time series so that they reach the
    target size.

    Parameters
    ----------
    sz : int
        Size of the output time series.

    Examples
    --------
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6]])
    array([[[0. ],
            [1.5],
            [3. ],
            [4.5],
            [6. ]]])
    """
    def __init__(self, sz):
        self.sz_ = sz

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.

        Parameters
        ----------
        X
            Ignored

        Returns
        -------
        self
        """
        return self

    def _transform_unit_sz(self, X):
        n_ts, sz, d = X.shape
        X_out = numpy.empty((n_ts, self.sz_, d))
        for i in range(X.shape[0]):
            X_out[i] = numpy.nanmean(X[i], axis=0, keepdims=True)
        return X_out

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        X_ = to_time_series_dataset(X)
        if self.sz_ == 1:
            return self._transform_unit_sz(X_)
        n_ts, sz, d = X_.shape
        equal_size = check_equal_size(X_)
        X_out = numpy.empty((n_ts, self.sz_, d))
        for i in range(X_.shape[0]):
            xnew = numpy.linspace(0, 1, self.sz_)
            if not equal_size:
                sz = ts_size(X_[i])
            for di in range(d):
                f = interp1d(numpy.linspace(0, 1, sz), X_[i, :sz, di],
                             kind="slinear")
                X_out[i, :, di] = f(xnew)
        return X_out


class TimeSeriesScalerMinMax(TransformerMixin, TimeSeriesBaseEstimator):
    """Scaler for time series. Scales time series so that their span in each
    dimension is between ``min`` and ``max`` where ``value_range=(min, max)``.

    Parameters
    ----------
    value_range : tuple (default: (0., 1.))
        The minimum and maximum value for the output time series.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

        NaNs within a time series are ignored when calculating min and max.

    Examples
    --------
    >>> TimeSeriesScalerMinMax(value_range=(1., 2.)).fit_transform([[0, 3, 6]])
    array([[[1. ],
            [1.5],
            [2. ]]])
    >>> TimeSeriesScalerMinMax(value_range=(1., 2.)).fit_transform(
    ...     [[numpy.nan, 3, 6]]
    ... )
    array([[[nan],
            [ 1.],
            [ 2.]]])
    """
    def __init__(self, value_range=(0., 1.)):
        self.value_range = value_range

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.

        Parameters
        ----------
        X
            Ignored

        Returns
        -------
        self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)
        self._X_fit_dims = X.shape
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):
        """Will normalize (min-max) each of the timeseries. IMPORTANT: this
        transformation is completely stateless, and is applied to each of
        the timeseries individually.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset.
        """
        value_range = self.value_range

        if value_range[0] >= value_range[1]:
            raise ValueError("Minimum of desired range must be smaller"
                             " than maximum. Got %s." % str(value_range))

        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)
        min_t = numpy.nanmin(X_, axis=1)[:, numpy.newaxis, :]
        max_t = numpy.nanmax(X_, axis=1)[:, numpy.newaxis, :]
        range_t = max_t - min_t
        range_t[range_t == 0.] = 1.
        nomin = (X_ - min_t) * (value_range[1] - value_range[0])
        X_ = nomin / range_t + value_range[0]
        return X_

    def _more_tags(self):
        return {'allow_nan': True}


class TimeSeriesScalerMeanVariance(TransformerMixin, TimeSeriesBaseEstimator):
    """Scaler for time series. Scales time series so that their mean (resp.
    standard deviation) in each dimension is
    mu (resp. std).

    Parameters
    ----------
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

        NaNs within a time series are ignored when calculating mu and std.

    Examples
    --------
    >>> TimeSeriesScalerMeanVariance(mu=0.,
    ...                              std=1.).fit_transform([[0, 3, 6]])
    array([[[-1.22474487],
            [ 0.        ],
            [ 1.22474487]]])
    >>> TimeSeriesScalerMeanVariance(mu=0.,
    ...                              std=1.).fit_transform([[numpy.nan, 3, 6]])
    array([[[nan],
            [-1.],
            [ 1.]]])
    """
    def __init__(self, mu=0., std=1.):
        self.mu = mu
        self.std = std

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.

        Parameters
        ----------
        X
            Ignored

        Returns
        -------
        self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)
        self._X_fit_dims = X.shape
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)
        mean_t = numpy.nanmean(X_, axis=1, keepdims=True)
        std_t = numpy.nanstd(X_, axis=1, keepdims=True)
        std_t[std_t == 0.] = 1.

        X_ = (X_ - mean_t) * self.std / std_t + self.mu

        return X_

    def _more_tags(self):
        return {'allow_nan': True}


class TimeSeriesScaleMeanMaxVariance(TimeSeriesScalerMeanVariance):
    """Scaler for time series. Scales time series so that their mean (resp.
    standard deviation) in the signal with the max amplitue  is
    mu (resp. std). The scaling relationships between each signal are preserved
    This is supplement to the TimeSeriesScalerMeanVariance method

    Parameters
    ----------
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

        NaNs within a time series are ignored when calculating mu and std.
    """

    def transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)
        mean_t = numpy.nanmean(X_, axis=1, keepdims=True)
        std_t = numpy.nanstd(X_, axis=1, keepdims=True)
        # retain the scaling relation cross the signals,
        # the max std_t is set to self.std
        max_std = max(std_t)
        if max_std ==0.: max_std = 1
        X_ = (X_ - mean_t) * self.std / max_std + self.mu

        return X_

    def _more_tags(self):
        return {'allow_nan': True, '_skip_test': True}



