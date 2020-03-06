import numpy as np

from src.classifier.riemannianClassifier.classifier import riemannianClassifier

clfpath = "../savedFilters/test"
clf = riemannianClassifier(savePath=clfpath)
clf.load_self()

def printWindowCallback(window):
    print(window.shape)

def printClassifierProbaCallback(window):
    window = np.asarray([window])
    print(window.shape)
    proba = clf.predict_proba(window)
    print(proba)