"""
The :mod:`tslearn.datasets` module provides simplified access to standard time
series datasets.
"""

import numpy
import zipfile
import tempfile
import shutil
import os
import csv
import warnings
from urllib.request import urlretrieve

from tslearn.utils import _load_arff_uea, _load_txt_uea

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def extract_from_zip_url(url, target_dir=None, verbose=False):
    """Download a zip file from its URL and unzip it.

    A `RuntimeWarning` is printed on failure.

    Parameters
    ----------
    url : string
        URL from which to download.
    target_dir : str or None (default: None)
        Directory to be used to extract unzipped downloaded files.
    verbose : bool (default: False)
        Whether to print information about the process (cached files used, ...)

    Returns
    -------
    str or None
        Directory in which the zip file has been extracted if the process was
        successful, None otherwise
    """
    fname = os.path.basename(url)
    tmpdir = tempfile.mkdtemp()
    local_zip_fname = os.path.join(tmpdir, fname)
    urlretrieve(url, local_zip_fname)
    os.makedirs(target_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(local_zip_fname, "r") as f:
            f.extractall(path=target_dir)
        if verbose:
            print("Successfully extracted file %s to path %s" %
                  (local_zip_fname, target_dir))
        return target_dir
    except zipfile.BadZipFile:
        warnings.warn("Corrupted or missing zip file encountered, aborting",
                      category=RuntimeWarning)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def in_file_string_replace(filename, old_string, new_string):
    """String replacement within a text file. It is used to fix typos in
    downloaded csv file.

    The code was modified from "https://stackoverflow.com/questions/4128144/"

    Parameters
    ----------
    filename : str
        Path to the file where strings should be replaced
    old_string : str
        The string to be replaced in the file.
    new_string : str
        The new string that will replace old_string
    """
    with open(filename) as f:
        s = f.read()

    with open(filename, 'w') as f:
        s = s.replace(old_string, new_string)
        f.write(s)


class UCR_UEA_datasets:
    """A convenience class to access UCR/UEA time series datasets.

    When using one (or several) of these datasets in research projects, please
    cite [1]_.

    This class will attempt to recover from some known misnamed files, like the
    `StarLightCurves` dataset being provided in `StarlightCurves.zip` and
    alike.

    Parameters
    ----------
    use_cache : bool (default: True)
        Whether a cached version of the dataset should be used in
        :meth:`~load_dataset`, if one is found.
        Datasets are always cached upon loading, and this parameter only
        determines whether the cached version shall be refreshed upon loading.

    Notes
    -----
        Downloading dataset files can be time-consuming, it is recommended
        using `use_cache=True` (default) in order to only experience
        downloading time once per dataset and work on a cached version of the
        datasets afterward.

    See Also
    --------
    Class :class:`CachedDatasets`
        Provides distinct selected datasets for offline use.

    References
    ----------
    .. [1] A. Bagnall, J. Lines, W. Vickers and E. Keogh, The UEA & UCR Time
       Series Classification Repository, www.timeseriesclassification.com
    """
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self._data_dir = os.path.expanduser(os.path.join(
            "~", ".tslearn", "datasets", "UCR_UEA"
        ))
        os.makedirs(self._data_dir, exist_ok=True)

        try:
            url_multivariate = ("https://www.timeseriesclassification.com/"
                                "Downloads/Archives/summaryMultivariate.csv")
            self._list_multivariate_filename = os.path.join(
                self._data_dir, os.path.basename(url_multivariate)
            )
            urlretrieve(url_multivariate, self._list_multivariate_filename)
            url_baseline = ("https://www.timeseriesclassification.com/"
                            "singleTrainTest.csv")
            self._baseline_scores_filename = os.path.join(
                self._data_dir, os.path.basename(url_baseline))
            urlretrieve(url_baseline, self._baseline_scores_filename)

            # fix typos in that CSV to match with the name in the download link
            in_file_string_replace(self._baseline_scores_filename,
                                   "CinCECGtorso", "CinCECGTorso")
            in_file_string_replace(self._baseline_scores_filename,
                                   "StarlightCurves", "StarLightCurves")
        except Exception:
            self._baseline_scores_filename = None

        self._ignore_list = ["Data Descriptions"]
        # File names for datasets for which it is not obvious
        # key: from timeseriesclassification.com, value: right dataset name
        self._filenames = {
            "AtrialFibrillation": "AtrialFibrilation",
            "CinCECGtorso": "CinCECGTorso",
            "MixedShapes": "MixedShapesRegularTrain",
            "NonInvasiveFetalECGThorax1": "NonInvasiveFatalECGThorax1",
            "NonInvasiveFetalECGThorax2": "NonInvasiveFatalECGThorax2",
            "StarlightCurves": "StarLightCurves"
        }

    def baseline_accuracy(self, list_datasets=None, list_methods=None):
        """Report baseline performances as provided by UEA/UCR website (for
        univariate datasets only).

        Parameters
        ----------
        list_datasets: list or None (default: None)
            A list of strings indicating for which datasets performance should
            be reported.
            If None, performance is reported for all datasets.
        list_methods: list or None (default: None)
            A list of baselines methods for which performance should be
            reported.
            If None, performance for all baseline methods is reported.

        Returns
        -------
        dict
            A dictionary in which keys are dataset names and associated values
            are themselves dictionaries that provide accuracy scores for the
            requested methods.

        Examples
        --------
        >>> uea_ucr = UCR_UEA_datasets()
        >>> dict_acc = uea_ucr.baseline_accuracy(
        ...         list_datasets=["Adiac", "ChlorineConcentration"],
        ...         list_methods=["C45"])
        >>> len(dict_acc)
        2
        >>> dict_acc["Adiac"]  # doctest: +ELLIPSIS
        {'C45': 0.542199...}
        >>> all_dict_acc = uea_ucr.baseline_accuracy()
        >>> len(all_dict_acc)
        85
        """
        with open(self._baseline_scores_filename, "r") as f:
            d_out = dict()
            for perfs_dict in csv.DictReader(f, delimiter=","):
                dataset_name = perfs_dict[""]
                if list_datasets is None or dataset_name in list_datasets:
                    d_out[dataset_name] = {}
                    for m in perfs_dict.keys():
                        if m != "" and (list_methods is None or m in list_methods):
                            try:
                                d_out[dataset_name][m] = float(perfs_dict[m])
                            except ValueError:  # Missing score case (score == "")
                                pass
            return d_out

    def list_univariate_datasets(self):
        """List univariate datasets in the UCR/UEA archive.

        Examples
        --------
        >>> l = UCR_UEA_datasets().list_univariate_datasets()
        >>> len(l)
        85

        Returns
        -------
        list of str:
            A list of the names of all univariate dataset namas.
        """
        with open(self._baseline_scores_filename, "r") as f:
            return [
                perfs_dict[""]  # get the dataset name
                for perfs_dict in csv.DictReader(f, delimiter=",")
            ]

    def list_multivariate_datasets(self):
        """List multivariate datasets in the UCR/UEA archive.

        Examples
        --------
        >>> l = UCR_UEA_datasets().list_multivariate_datasets()
        >>> "PenDigits" in l
        True

        Returns
        -------
        list of str:
            A list of the names of all multivariate dataset namas.
        """
        with open(self._list_multivariate_filename, "r") as f:
            return [
                infos_dict["Problem"]  # get the dataset name
                for infos_dict in csv.DictReader(f, delimiter=",")
            ]

    def list_datasets(self):
        """List datasets (both univariate and multivariate) available in the 
        UCR/UEA archive.

        Examples
        --------
        >>> l = UCR_UEA_datasets().list_datasets()
        >>> "PenDigits" in l
        True
        >>> "BeetleFly" in l
        True
        >>> "DatasetThatDoesNotExist" in l
        False

        Returns
        -------
        list of str:
            A list of names of all (univariate and multivariate) dataset namas.
        """
        return (self.list_univariate_datasets()
                + self.list_multivariate_datasets())

    def list_cached_datasets(self):
        """List datasets from the UCR/UEA archive that are available in cache.

        Examples
        --------
        >>> beetlefly = UCR_UEA_datasets().load_dataset("BeetleFly")
        >>> l = UCR_UEA_datasets().list_cached_datasets()
        >>> "BeetleFly" in l
        True
        """
        return [path for path in os.listdir(self._data_dir)
                if os.path.isdir(os.path.join(self._data_dir, path)) and
                path not in self._ignore_list]

    def load_dataset(self, dataset_name):
        """Load a dataset from the UCR/UEA archive from its name.

        On failure, `None` is returned for each of the four values and a
        `RuntimeWarning` is printed.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset. Should be in the list returned by
            `list_datasets`

        Returns
        -------
        numpy.ndarray of shape (n_ts_train, sz, d) or None
            Training time series. None if unsuccessful.
        numpy.ndarray of integers or strings with shape (n_ts_train, ) or None
            Training labels. None if unsuccessful.
        numpy.ndarray of shape (n_ts_test, sz, d) or None
            Test time series. None if unsuccessful.
        numpy.ndarray of integers or strings with shape (n_ts_test, ) or None
            Test labels. None if unsuccessful.

        Examples
        --------
        >>> data_loader = UCR_UEA_datasets()
        >>> X_train, y_train, X_test, y_test = data_loader.load_dataset(
        ...         "TwoPatterns")
        >>> X_train.shape
        (1000, 128, 1)
        >>> y_train.shape
        (1000,)
        >>> X_train, y_train, X_test, y_test = data_loader.load_dataset(
        ...         "CinCECGTorso")
        >>> X_train.shape
        (40, 1639, 1)
        >>> X_train, y_train, X_test, y_test = data_loader.load_dataset(
        ...         "PenDigits")
        >>> X_train.shape
        (7494, 8, 2)
        >>> assert (None, None, None, None) == data_loader.load_dataset(
        ...         "DatasetThatDoesNotExist")
        """
        dataset_name = self._filenames.get(dataset_name, dataset_name)
        full_path = os.path.join(self._data_dir, dataset_name)

        if not self._has_files(dataset_name) or not self.use_cache:
            # completely clear the target directory first, it will be created
            # by extract_from_zip_url if it does not exist
            try:
                # maybe it is a normal file (not a directory) for some obscure reason
                os.remove(full_path)
            except FileNotFoundError:
                pass  # nothing to do here, already deleted
            except IsADirectoryError:
                # we then need this function to recursively remove the directory:
                shutil.rmtree(full_path, ignore_errors=True)
            # else, actually raise the error!

            url = ("https://www.timeseriesclassification.com/Downloads/%s.zip"
                   % dataset_name)
            success = extract_from_zip_url(url, target_dir=full_path)
            if not success:
                warnings.warn("dataset \"%s\" could not be downloaded or "
                              "extracted" % dataset_name,
                              category=RuntimeWarning, stacklevel=2)
                return None, None, None, None

        try:
            # if both TXT and ARFF files are provided, the TXT versions are
            # used
            # both training and test data must be available in the same format
            if self._has_files(dataset_name, ext="txt"):
                X_train, y_train = _load_txt_uea(
                    os.path.join(full_path, dataset_name + "_TRAIN.txt")
                )
                X_test, y_test = _load_txt_uea(
                    os.path.join(full_path, dataset_name + "_TEST.txt")
                )
            elif self._has_files(dataset_name, ext="arff"):
                X_train, y_train = _load_arff_uea(
                    os.path.join(full_path, dataset_name + "_TRAIN.arff")
                )
                X_test, y_test = _load_arff_uea(
                    os.path.join(full_path, dataset_name + "_TEST.arff")
                )
            else:
                warnings.warn("dataset \"%s\" is not provided in either TXT "
                              "or ARFF format and thus could not be loaded"
                              % dataset_name,
                              category=RuntimeWarning, stacklevel=2)
                return None, None, None, None

            return X_train, y_train, X_test, y_test

        except Exception as exception:
            warnings.warn("dataset \"%s\" could be downloaded but not "
                          "parsed: %s" % (dataset_name, str(exception)),
                          category=RuntimeWarning, stacklevel=2)
            return None, None, None, None

    def _has_files(self, dataset_name, ext=None):
        """Determines whether some downloaded and unzipped dataset provides
        both training and test data in the given format.

        Parameters
        ----------
        dataset_name : str
            the name of the dataset
        ext : str or None
            the file extension without a dot, e.g `"txt"` or `"arff"`;
            if set to None (the default), `True` will be returned if either TXT
            or ARFF files are present

        Returns
        -------
        bool
            if there are both training and test files with the specified
            file extension
        """
        if ext is None:
            return (self._has_files(dataset_name, ext="txt") or
                    self._has_files(dataset_name, ext="arff"))
        else:
            dataset_name = self._filenames.get(dataset_name, dataset_name)
            full_path = os.path.join(self._data_dir, dataset_name)
            basename = os.path.join(full_path, dataset_name)
            return (os.path.exists(basename + "_TRAIN.%s" % ext) and
                    os.path.exists(basename + "_TEST.%s" % ext))

    def cache_all(self):
        """Cache all datasets from the UCR/UEA archive for later use."""
        for dataset_name in self.list_datasets():
            try:
                self.load_dataset(dataset_name)
            except Exception as exception:
                warnings.warn("Could not cache dataset \"%s\": %s"
                              % (dataset_name, str(exception)),
                              category=RuntimeWarning, stacklevel=2)


class CachedDatasets:
    """A convenience class to access cached time series datasets.

    Note, that these *cached datasets* are statically included into *tslearn*
    and are distinct from the ones in :class:`UCR_UEA_datasets`.

    When using the Trace dataset, please cite [1]_.

    References
    ----------
    .. [1] A. Bagnall, J. Lines, W. Vickers and E. Keogh, The UEA & UCR Time
       Series Classification Repository, www.timeseriesclassification.com
    """
    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__), ".cached_datasets")

    def list_datasets(self):
        """List cached datasets.

        Examples
        --------
        >>> _ = UCR_UEA_datasets().load_dataset("Trace")
        >>> cached = UCR_UEA_datasets().list_cached_datasets()
        >>> "Trace" in cached
        True

        Returns
        -------
        list of str:
            A list of names of all cached (univariate and multivariate) dataset
            namas.
        """
        return [fname[:fname.rfind(".")]
                for fname in os.listdir(self.path)
                if fname.endswith(".npz")]

    def load_dataset(self, dataset_name):
        """Load a cached dataset from its name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset. Should be in the list returned by
            :meth:`~list_datasets`.

        Returns
        -------
        numpy.ndarray of shape (n_ts_train, sz, d) or None
            Training time series. None if unsuccessful.
        numpy.ndarray of integers with shape (n_ts_train, ) or None
            Training labels. None if unsuccessful.
        numpy.ndarray of shape (n_ts_test, sz, d) or None
            Test time series. None if unsuccessful.
        numpy.ndarray of integers with shape (n_ts_test, ) or None
            Test labels. None if unsuccessful.

        Examples
        --------
        >>> data_loader = CachedDatasets()
        >>> X_train, y_train, X_test, y_test = data_loader.load_dataset(
        ...                                        "Trace")
        >>> print(X_train.shape)
        (100, 275, 1)
        >>> print(y_train.shape)
        (100,)

        Raises
        ------
        IOError
            If the dataset does not exist or cannot be read.
        """
        npzfile = numpy.load(os.path.join(self.path, dataset_name + ".npz"))
        X_train = npzfile["X_train"]
        X_test = npzfile["X_test"]
        y_train = npzfile["y_train"]
        y_test = npzfile["y_test"]
        return X_train, y_train, X_test, y_test
