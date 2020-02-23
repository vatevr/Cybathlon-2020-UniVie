from sklearn.base import BaseEstimator
import pyriemann
import numpy as np
from features import TSSF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import softmax
from sklearn.svm import SVC
import pickle


class riemannianClassifier(BaseEstimator):
    def __init__(
        self,
        metric="riemann",
        filtering=None,
        n_components=None,
        two_step_classifier=None,
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
        self.filtering = filtering
        self.secondClassifier = two_step_classifier

        if filtering == None:
            self.clf = pyriemann.classification.MDM(metric=self.metric)
        if filtering == "geodesic":
            self.clf = pyriemann.classification.FgMDM(metric=self.metric)
        if filtering == "TSSF":
            if two_step_classifier is None:
                self.clf = TSSF(ts_metric=self.metric, n_components=n_components)
            else:
                #two step classification is still buggy
                self.clf = make_pipeline(
                    TSSF(ts_metric=self.metric, n_components=n_components),
                    self.secondClassifier
                )

    def load_self(self) :
        clf_filename = "classifier.sav"
        le_filename = "le.sav"
        param_filename = "params.sav"
        self.clf = pickle.load(open(clf_filename, "rb"))
        self.le = pickle.load(open(le_filename, "rb"))
        params = pickle.load(open(param_filename, "rb"))
        self.filtering = params["filtering"]
        return self

    def save_self(self) :
        clf_filename = "classifier.sav"
        le_filename = "le.sav"
        param_filename = "params.sav"
        pickle.dump(self.clf, open(clf_filename, "wb"))
        pickle.dump(self.le, open(le_filename, "wb"))
        pickle.dump(self.get_params(), open(param_filename, "wb"))


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
        self.le = LabelEncoder()
        self.le.fit(y)
        y_encoded = self.le.transform(y)

        if self.filtering == "TSSF":
            self.clf.fit(X, y_encoded)
        else:
            cov = pyriemann.estimation.Covariances().fit_transform(X)
            self.clf.fit(cov, y_encoded)

        self.save_self()
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
        y = self.predict_proba(X)
        y = [0 if val[0] > val[1] else 1 for val in y]
        return self.le.inverse_transform(y)

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X: ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        
        Returns
        ----------
        y: ndarray, shape(n_trials, n_classes)
            class probabilities
        """
        if self.filtering == "TSSF":
            decision_vals = self.clf.decision_function(X)
            prob = softmax(np.array([[-val if val < 0 else 0 for val in decision_vals], [val if val > 0 else 0 for val in decision_vals]]).T)
           
        else:
            cov = pyriemann.estimation.Covariances().fit_transform(X)
            prob = self.clf.predict_proba(cov)
        return prob

    def score(self, X, y):
        y_pred = self.predict(X)
        return len([(a, b) for (a, b) in list(zip(y_pred, y)) if a == b]) / len(y)

