from Pipeline.src.classifier.riemannianClassifier.classifier import riemannianClassifier
import numpy as np
from pylsl import StreamInfo, StreamOutlet

clfpath = "../savedFilters/test"
clf = riemannianClassifier(savePath=clfpath)
clf.load_self()

#Set LSL
LSLchannels = 4
LSLinterval = 10
LSLinfo = StreamInfo('Feedback', 'VIS', LSLchannels, LSLinterval, 'float32', 'myuid34234')
LSLoutlet = StreamOutlet(LSLinfo)

def printWindowCallback(window):
    print(window.shape)

def printClassifierProbaCallback(window):
    window = np.asarray([window])
    print(window.shape)
    proba = clf.predict_proba(window)
    print(proba)

def sendToVisTest(window) :
    window = np.asarray([window])
    print(window.shape)
    proba = clf.predict_proba(window)
    print(proba)
    LSLoutlet.push_sample(proba[0])

