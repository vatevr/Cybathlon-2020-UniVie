# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:52:43 2019

@author: Matthaeus Hilpold

Largely based on example code from the MNE library : https://github.com/mne-tools/mne-python/blob/1325c1abffa0db8cf57c9cd6410a4c92fcb0586c/examples/decoding/plot_decoding_csp_eeg.py
by Martin Billinger, license: BSD (3-clause)

And on code from the sklearn library.
"""

import numpy as np
import matplotlib.pyplot as plt
import amplitude_extraction as sp
import topomap_plot as tp

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier as OVR #this meta-classifier can be used for our classification at CYBATHLON
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, cross_val_score, learning_curve #various algorithms to cross validate and evaluate. Maybe RepeatedKFold as well? 

from mne import Epochs, pick_types, events_from_annotations
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from FBCSP import FBCSP


def plot_learning_curve(estimator, X, y, cv) :
    """
    :param estimator : sklearn-estimator
    :param X : List/numpy-array data
    :param y : List/numpy-array class labels
    :param cv : cross-validation or pipeline
    """
    train_sizes, train_scores, test_scores = learning_curve(clf, epochs_data_train, labels, cv=cv, verbose=0)
    
    #mean and standard deviation for train scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    #mean and standard deviation for test scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    #plot learning curve
    
    # Draw lines
    plt.plot(train_sizes, train_scores_mean, '--', color="#ff0000",  label="Training score")
    plt.plot(train_sizes, test_scores_mean, color="#0000ff", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="#DDDDDD")
    
    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Cross Validation Accuracy Score"), plt.legend(loc="best")
    #plt.tight_layout()
    plt.show()
    
def plot_scores_over_time(w_start, w_length, cv_split, epochs, epochs_train, labels, sfreq) :   
    scores_windows = []
    
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]
    
        X_train = fbcsp.fit_transform(epochs_train[train_idx], y_train)
        X_test = fbcsp.transform(epochs_train[test_idx])
        
        X_train_mne = csp.fit_transform(epochs_train[train_idx], y_train)
        X_test_mne = csp.transform(epochs_train[test_idx])
        # fit classifier
        svm.fit(X_train, y_train)
        print(X_train.shape)
        print(X_train_mne.shape)
        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = fbcsp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(svm.score(X_test, y_test))
        scores_windows.append(score_this_window)
        
        # Plot scores over time
        w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
        
    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()

"""
def plot_csp(epochs, epochs_data, y) :
    
    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)
    
    layout = read_layout('EEG1005')
    csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',units='Patterns (AU)', size=1.5)
"""



if __name__ == '__main__':
    
    #load data
    #mne.io.read_raw_brainvision() skip this step, since we are using a sample dataset
    
    
    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.
    tmin, tmax = -1., 4.    #default is -0.2, 0.5
    event_id = dict(left=1, right=2)
    subject = 1
    runs = [4, 8, 12] # motor imagery: right hand vs. left hand
    
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    
    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))
    
    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    
    #picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True) #picks to pick specific channels
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2
    
    # Cross validation
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    
    cv = RepeatedKFold(10, 5) # repeated Kfold. Number_of_folds x Number_of_Repetitions
    cv_split = cv.split(epochs_data_train)
    
    # Classifier. Sci-kit's multi-class SVC uses one-versus-one approach, but we need one-versus-rest. Therefore LinearSVC is used.
    penalty='l2' #L2 norm is used for penalty
    svm = LinearSVC(penalty=penalty)
    csp = CSP()
    fbcsp = FBCSP(filter_target='epoched', concatenate=False, avg_band=True)
    fbcsp.fit(epochs_data_train, labels)
    
    custom_csp_result = fbcsp.transform(epochs_data_train)
    
    
    #theirresult = csp.fit_transform(epochs_data_train, labels)
    
    
    #scores = cross_val_score(svm, epochs_data_train, labels, cv=cv, n_jobs=1, verbose=0)
    
    sfreq = raw.info['sfreq']
    w_length = int(sfreq * 0.5)   # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
    
    
    #score over time
    plot_scores_over_time(w_start, w_length, cv_split, epochs, epochs_data_train, labels, sfreq)
    
    #CSP
    #plot_csp(epochs, epochs_data, labels)
    
    """
    # Sklearn Pipelines can be useful, if the same steps are always performed. In our case it will be pre-processing -> signal-processing -> filter-bank -> csp -> [feature-selection] -> classification.
    # Later it can potentially allow us to grid-search parameters for every step in a unified way.
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, verbose=0) #estimator, data to be fit, classes, cross validator, number of CPU's        
    
 

    sfreq = raw.info['sfreq']
    w_length = int(sfreq * 0.5)   # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
    
    
    #score over time
    plot_scores_over_time(w_start, w_length, cv_split, epochs, epochs_data_train, labels, sfreq)
    
    # learning curve   
    plot_learning_curve(clf, epochs_data_train, labels, cv)
    
    #For our sample size of 40 for two classes, an accuracy of at least 67.5% needs to be reached for a p-value below 0.01
    #according to: https://www.ncbi.nlm.nih.gov/pubmed/25596422  
    print("Classification accuracy: %f" % np.mean(scores))
    """