"""
===========================================================================
Source code for tangent space spatial filter (TSSF)

Author: Jiachen XU <jiachen.xu.94@gmail.com> 

Last Update: 2019-11-18
===========================================================================

To use this function, TSSF() class can be merged into moabb.pipelines.features
"""
import numpy as np

from scipy import linalg

from pyriemann.utils.base import logm
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class TSSF(BaseEstimator, TransformerMixin):

    """ Tangent Space Spatial Filter framework


    Parameters
    ----------
    clf_str : 'LR' | 'LDA' | 'SVM', default 'SVM'
        Linear classifier on the tangent space (the first classifer in [1]).
        'LR': L1 regularized linear regressor (LASSO).
        'LDA': Fisher Linear Discriminant Analysis classifier.
        'SVM': L2 regularized Support Vector Machine using grid search.
    func : 'clf' | 'filter' | 'pattern', default 'clf'
        'clf': To use the classification function of TSSF
            (incl. both One-step and Two-step classification [1])
        'filter': To plot the derived spatial filters
        'pattern': To plot the associated spatial patterns of TSSF
    n_components: None | int, default None
        Number of used filters components.
        If n_components is not None, then use the specified number of TSSF components.
        Otherwise, defaults to n_channels
    comp_order: None | ndarray, shape (n_components, 1), default None
        If comp_order is not None, then use the specified order of components.
        Otherwise, the first n_components TSSF will be selected by default.
    decomp: 'ED' | 'GED', default 'GED'
        which kind of decomposition do you want to use to generate the filters
    cov_reg : string, default 'scm'
        Covariance estimator on the sklearn toolbox.
        Other options please refer to sklearn.covariance.
        !!! Please carefully choose diagonal-loading based estimator
            (Sec. V.E.1 of [1]).
    logvar : bool, default True
        To use log-variance features, i.e., Log-var in Table III.1 of [1],
            please set to True
        To use logarithm-covariance features, i.e., Diag. log-cov or Log-cov
            in Table III.1 of [1], please set to False
    ts_metric : string, default 'riemann'
        Riemannian metric for computing the manifold mean.
        Other options please refer to pyriemann.mean_covariance()

    Attributes
    ----------
    ts : object,
        Fitted tangent space for data with full dimensionality
    ts_proj : object,
        Fitted tangent space for the filtered data (i.e., with reduced
        dimensionality)
    filters_ :  ndarray, shape (n_channels, n_components)
        If fit, the TSSF components used to decompose the data, else None.
    patterns_ : ndarray, shape (n_channels, n_components)
        If fit, the TSSF patterns used to restore EEG signals, else None.
    beta : ndarray, shape (n_components, )
        Coefficient for one-step classification

    References
    ----------

    [1] J. Xu, M. Grosse-Wentrup, and V. Jayaram. Tangent space spatial filters
        for interpretable and efficient Riemannian classification. In: (2019).
        arXiv: 1909.10567.

    [2] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J. D.,
        Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight
        vectors of linear models in multivariate neuroimaging. Neuroimage,
        87, 96-110.
    """

    def __init__(self, clf_str='SVM', func='clf', n_components=None,
                 comp_order=None, decomp='GED', cov_reg='scm', logvar=True,
                 ts_metric='riemann'):

        self.clf_str = clf_str
        self.func = func
        self.n_components = n_components
        self.comp_order = comp_order
        self.decomp = decomp
        self.cov_reg = cov_reg
        self.logvar = logvar
        self.ts_metric = ts_metric

        self.ts = TangentSpace(metric=ts_metric)
        self.cov_clf = Covariances(estimator=cov_reg)

        self.best_clf = None

    def filter_generator(self, W, Cmean):
        if self.decomp == "ED":
            d, V = linalg.eigh(W)
        elif self.decomp == "GED":
            d, V = linalg.eigh(W, Cmean)
        else:
            raise ValueError("Wrong decomposition! Either ED or GED")

        # Ordering based on log-eigenvalues of GED
        inds = np.argsort(np.abs(np.log(d)))[::-1]

        return V[:, inds], d[inds]

    def TS_classification(self, X, y):
        if self.clf_str == 'LR':
            self.clf_ = LR(penalty='l1')
        elif self.clf_str == 'LDA':
            self.clf_ = LDA()
        elif self.clf_str == 'SVM':
            parameters = {'C': np.logspace(-2, 2, 10)}
            self.clf_ = GridSearchCV(SVC(kernel='linear'), parameters)
        else:
            raise ValueError("Wrong classifier! The supported "
                             "classifiers are: LR(), LDA(), SVM() ")

        # Extract the weight vectors on the tangent space
        # and then projected them onto Riemannian manifold
        if isinstance(self.clf_, GridSearchCV):
            all_clfs = self.clf_.fit(X, y)
            self.best_clf = all_clfs.best_estimator_
            ts_coef = self.best_clf.coef_
        else:
            self.clf_.fit(X, y)
            self.best_clf = self.clf_
        ts_coef = self.best_clf.coef_
        if hasattr(self.best_clf, 'intercept_'):
            self.bias = self.best_clf.intercept_

        self.y_ts_true = ts_coef.dot(X.T)
        #  The reshape function is for the compatibility with Pyriemann pkg.
        cov_mat_coef = self.ts.inverse_transform(np.reshape(ts_coef, [1, -1]))

        return ts_coef, cov_mat_coef[0]

    def fit(self, X, y):
        assert len(np.unique(y)) == 2, "Only works with binary classification"
        self.cov_all = self.cov_clf.transform(X)
        X_ = self.ts.fit_transform(self.cov_all, y)
        cov_mat_mean_ = self.ts.reference_

        # --------------- Classifying on the TS --------------------
        self.w_, self.C_w = self.TS_classification(X_, y)

        # --------------- Extract SF from classifier --------------------
        eigen_vectors, ori_eig = self.filter_generator(self.C_w, cov_mat_mean_)

        # ---------------------- Selecting the SF -------------------------
        #  Select filter components based on given order or the first K comp.
        if self.n_components is not None:
            if self.comp_order is not None:
                filters_ = eigen_vectors[:, self.comp_order]
            else:
                filters_ = eigen_vectors[:, :self.n_components]
        else:
            filters_ = eigen_vectors

        # ---------------------- Applying the SF -------------------------
        if self.func == 'clf':
            self.coef_ = filters_.T

            # Find the new reference point on the low-dim TS
            from pyriemann.tangentspace import TangentSpace as ts_proj
            X_fil = np.asarray([np.dot(self.coef_, epoch) for epoch in X])
            X_cov = self.cov_clf.transform(X_fil)
            self.ts_proj = ts_proj(metric=self.ts_metric).fit(X_cov)

        elif self.func == 'pattern':
            if self.n_components is None:
                #  Simplest  way to derive patterns but requires squire size
                patterns_ = linalg.pinv2(filters_)
            else:
                #  Derive patterns for SF without full rank [1]
                sigma_X = cov_mat_mean_
                sigma_S = np.dot(filters_.T, np.dot(sigma_X, filters_))
                patterns_ = np.dot(
                    sigma_X, np.dot(filters_, linalg.pinv2(sigma_S)))

            self.patterns_ = patterns_

        elif self.func == 'filter':
            self.filters_ = filters_
        else:
            raise ValueError("Valid string for func is either 'filter' or 'pattern'"
                             ", for classification please use 'clf'.")

        self.reg_clf = clone(self.best_clf).fit(X_, y)

        self.ori_eig = ori_eig
        self.beta = np.log(ori_eig[:self.n_components])

        return self

    def transform(self, X):
        if self.func == 'clf':
            X_fil = np.asarray([np.dot(self.coef_, epoch) for epoch in X])
            X_cov = self.cov_clf.transform(X_fil)
            if not self.logvar:
                X_ts = self.ts_proj.transform(X_cov)
            else:
                X_ts = np.asarray([np.log(np.diag(x)) for x in X_cov])
            return X_ts
        elif self.func == 'pattern':
            data_re = {'matrix': self.patterns_, 'data': X}
            return data_re
        elif self.func == 'filter':
            data_re = {'matrix': self.filters_, 'data': X}
            return data_re
        else:
            raise ValueError("Valid string for func is either 'filter' or 'pattern'"
                             ", for classification please use 'clf'.")

    def decision_function(self, X):
        X_filt = np.asarray([np.dot(self.coef_, epoch) for epoch in X])
        if not self.logvar:
            X_ts = np.asarray([np.diag(logm(
                self.cov_clf.transform(x[None, ...])[0])) for x in X_filt])
        else:
            X_ts = np.asarray([np.log(np.var(x, axis=1)) for x in X_filt])
        return X_ts.dot(self.beta)



