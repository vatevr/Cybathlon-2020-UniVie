#%%
from classifier import riemannianClassifier
from moabb.datasets import BNCI2014001, BNCI2014004, BNCI2015004
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
import warnings

warnings.simplefilter("ignore")

MI = MotorImagery(fmin=8, fmax=32)
datasets = [BNCI2014004(), BNCI2014001(), BNCI2015004()]
dataset_names = ["BNCI2014004", "BNCI2014001", "BNCI2015004"]

X, y, meta = MI.get_data(datasets[1], [1])


#%%
n_components = 5
classifiers = {
    "Default": riemannianClassifier(n_components=n_components),
    #"Geodesic Filtering": riemannianClassifier(filtering="geodesic"),
    "TSSF one step": riemannianClassifier(filtering="TSSF", n_components=n_components),
    "TSSF two steps": riemannianClassifier(filtering="TSSF", two_step_classifier=SVC(), n_components=n_components),
}

#%%
for clf in classifiers.values():
    clf.fit(X, y)
    print(cross_val_score(clf, X, y, scoring="roc_auc"))

#%%
""" X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
) """



#%%
