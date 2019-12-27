# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 17:58:27 2019

@author: Matthaeus Hilpold
"""


from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, cross_val_score, learning_curve #various algorithms to cross validate and evaluate. Maybe RepeatedKFold as well? 

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from FBCSP import FBCSP
from mne import Epochs, pick_types, events_from_annotations
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

    
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
svm_mne = LinearSVC(penalty=penalty)
csp = CSP()
fbcsp = FBCSP(filter_target='epoched', concatenate=False, avg_band=True)
fbcsp.fit(epochs_data_train, labels)

scaler = StandardScaler()

custom_csp_result = fbcsp.transform(epochs_data_train)

#scaler.fit(custom_csp_result)

#custom_csp_result = scaler.transform(custom_csp_result)

#svm.fit(custom_csp_result[:-1, :], labels[:-1])

sfreq = raw.info['sfreq']
w_length = int(sfreq * 1+1)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []
scores_windows_mne = []

for train_idx, test_idx in cv_split:
      y_train, y_test = labels[train_idx], labels[test_idx]
      
      X_train = fbcsp.fit_transform(epochs_data_train[train_idx], y_train)
      X_test = fbcsp.transform(epochs_data_train[test_idx])
      
      X_train_mne = csp.fit_transform(epochs_data_train[train_idx], y_train)
      X_test_mne = csp.transform(epochs_data_train[test_idx])
      # fit classifier
      svm_mne.fit(X_train_mne, y_train)
      svm.fit(X_train, y_train)
      print(X_train.shape)
      print(X_train_mne.shape)
      print('X_train shape: ' + str(X_train.shape))# running classifier: test classifier on sliding window
      print('X_train_mne shape: ' + str(X_train_mne.shape))# running classifier: test classifier on sliding window
      print('X_test shape: ' + str(X_test.shape))# running classifier: test classifier on sliding window
      print('X_test_mne shape: ' + str(X_test_mne.shape))# running classifier: test classifier on sliding window
      
      score_this_window_mne = []
      for n in w_start:
          X_test_mne = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
          score_this_window_mne.append(svm_mne.score(X_test_mne, y_test))
      scores_windows_mne.append(score_this_window_mne)
      
      score_this_window = []
      for n in w_start:
          X_test = fbcsp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
          score_this_window.append(svm.score(X_test, y_test))
      scores_windows.append(score_this_window)
      
      # Plot scores over time
      w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
       
plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score custom FBCSP')
plt.plot(w_times, np.mean(scores_windows_mne, 0), label='Score mne CSP')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()