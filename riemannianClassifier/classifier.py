from sklearn.base import BaseEstimator
import pyriemann
from features import TSSF
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


class riemannianClassifier(BaseEstimator):
    def __init__(
        self,
        metric="riemann",
        filtering=None,
        n_components=None,
        two_step_classifier=SVC(),
    ):
        """
        Parameters
        ----------
        metric: string, default "riemann"
            choose between "riemann" and "logeuclid"
        filtering : string, optional
            choose between "geodesic", "TSSF" or None (no filtering)
        n_components: int, optional
            number of filter components that are to be applied to the input data
        two_step_classifier : classifier, optional
            classifier that is to be used for the two step classification using TSSF
        """
        self.metric = metric
        if filtering == None:
            self.clf = pyriemann.classification.MDM(metric=self.metric)
        if filtering == "geodesic":
            self.clf = pyriemann.classification.FgMDM(metric=self.metric)
        if filtering == "TSSF":
            self.clf = make_pipeline(
                TSSF(ts_metric=self.metric, n_components=n_components),
                two_step_classifier
            )

        self.filtering = filtering
        self.secondClassifier = two_step_classifier

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        y: ndarray, shape (n_trials, 1)
            labels
        
        Returns
        ----------
        self: riemannianClassifier instance
        """

        if self.filtering == "TSSF":
            self.clf.fit(X, y)
        else:
            cov = pyriemann.estimation.Covariances().fit_transform(X)
            self.clf.fit(cov, y)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X: ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        
        Returns
        ----------
        y: ndarray
            predicted labels
        """
        if self.filtering == "TSSF":
            y = self.clf.predict(X)
        else:
            cov = pyriemann.estimation.Covariances().fit_transform(X)
            y = self.clf.predict(cov)
        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        return len([(a, b) for (a, b) in list(zip(y_pred, y)) if a == b]) / len(y)

