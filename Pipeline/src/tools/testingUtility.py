from Pipeline.src.classifier.riemannianClassifier.classifier import riemannianClassifier
import numpy as np

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