from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, cross_val_score, learning_curve #various algorithms to cross validate and evaluate. Maybe RepeatedKFold as well? 
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from FBCSP import FBCSP
from mne import Epochs, pick_types, events_from_annotations
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle
import amplitude_extraction as sp #Melanie Balaz' amplitude_extraction script


    
def save_filters (filters, filename='csp_filters.sav') :
    pickle.dump(filters, open(filename, 'wb'))
    
def save_classifier(classifier, filename='classifier.sav') :
    pickle.dump(classifier, open(filename, 'wb'))
    
def load_filters (filename='csp_filters.sav') :
    filters = pickle.load(open(filename, 'rb'))
    return filters
    
def load_classifier (filename='classifier.sav') :
    classifier = pickle.load(open(filename, 'rb'))
    return classifier

def score_window(data, filterpath='csp_filters.sav', classifierpath='classifier.sav') :
    """
    :param data : numpy array of one window of eeg data. Shape: channels * samples
    """
    classifier = load_classifier()
    filters = load_filters()
    data = np.asarray([data])      #adding a dimension to save some if statements in fbcsp
    data = filters.transform(data)
    prediction = classifier.predict([data[0]])
    return prediction


def compare_CSP_with_FBCSP(raw_data) :
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True) #picks to pick specific channels
    labels = epochs.events[:, -1] - 1 #turning it into 0 and 1
    
    epochs = epochs.copy().crop(tmin=1., tmax=2.)
    epochs_data = epochs.get_data()
    
    
    csp = CSP();
    fbcsp = FBCSP(filter_target='epoched', concatenate=False, avg_band=True);
    
    fbcsp_data = fbcsp.fit_transform(epochs_data, labels)
    csp_data = csp.fit_transform(epochs_data, labels)
    
    print("First, CSP and FBCSP are fit on the same epoched data equally")
    print("after the transformation, the shapes are:")
    print(fbcsp_data.shape)
    print(csp_data.shape)
    print("FBCSP has 16 components because amplitude extraction bandpass filters the signal")
    
    print("comparing means and std deviations: ")
    print("Mean FBCSP: " + str(np.mean(fbcsp_data)))
    print("Mean CSP: " + str(np.mean(csp_data)))
    print("Standard deviation FBCSP: " + str(np.std(fbcsp_data)))
    print("Standard deviation CSP: " + str(np.std(csp_data)))
    
def test(raw) :
    
        
    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
        
    #picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True) #picks to pick specific channels
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 1 #turning it into 0 and 1
        
    # Cross validation
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
        
    
    cv = RepeatedKFold(10, 5) # repeated Kfold. Number_of_folds x Number_of_Repetitions
    cv_split = cv.split(epochs_data_train)
        
    # Classifier. Sci-kit's multi-class SVC uses one-versus-one approach, but we need one-versus-rest. Therefore LinearSVC is used.
    penalty='l2' #L2 norm is used for penalty for SVM
    svm = LinearSVC(penalty=penalty)
    svm_mne = LinearSVC()
    csp = CSP()
    fbcsp = FBCSP(filter_target='epoched', concatenate=False, avg_band=True)
    #fbcsp.fit(epochs_data_train, labels)
    
    #scaler = StandardScaler()
    
    #custom_csp_result = fbcsp.transform(epochs_data_train)
    
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
    
    
    clf = Pipeline([('CSP', fbcsp), ('SVM', svm)])
        
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
    
    save_classifier(svm)
    save_filters(fbcsp)

    

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
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True) #picks to pick specific channels
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 1 #turning it into 0 and 1
    epochs_data_train = epochs_train.get_data()

    #test
    test(raw)
    compare_CSP_with_FBCSP(raw) 
    
    print(score_window(epochs_data_train[3]))
    print("should be")
    print(labels[3])