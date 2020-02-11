#%%
from classifier import riemannianClassifier
from moabb.datasets import BNCI2014001, BNCI2014004, BNCI2015004
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
import warnings

warnings.simplefilter("ignore")

MI = MotorImagery(fmin=8, fmax=32)
datasets = [BNCI2014004(), BNCI2014001(), BNCI2015004()]
dataset_names = ["BNCI2014004", "BNCI2014001", "BNCI2015004"]

X, y, meta = MI.get_data(datasets[0], [1])


#%%
classifiers = {
    "Default": riemannianClassifier(),
    "Geodesic Filtering": riemannianClassifier(filtering="geodesic"),
    "TSSF two steps": riemannianClassifier(filtering="TSSF"),
}

#%%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)
for clf in classifiers.values():
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


#%%
