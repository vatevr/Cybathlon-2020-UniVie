from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, cross_val_score, learning_curve #various algorithms to cross validate and evaluate. Maybe RepeatedKFold as well? 
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from classifier import classifier

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from FBCSP import FBCSP
from FBCSP import Preproc
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
    fbcsp = FBCSP(filter_target='epoched', method='avg_power');
    
    preproc = Preproc()
    X_preprocd = preproc.bpfilter_data(epochs_data)
    
    fbcsp_data = fbcsp.fit_transform(X_preprocd, labels)
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
    
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    
    preproc = Preproc(fs=161, windowlength=1, bands='all')
    raw_bpfiltered = preproc.preproc(raw.get_data(), events.T[0])
    
    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    
    
        
    #picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True) #picks to pick specific channels
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 1 #turning it into 0 and 1
        
    # Cross validation
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
        
    
    cv = RepeatedKFold(10, 5) # repeated Kfold. Number_of_folds x Number_of_Repetitions
    cv_split = cv.split(epochs_data_train)
        
    # Classifier. Sci-kit's multi-class SVC uses one-versus-one approach, but we need one-versus-rest. Therefore LinearSVC is used.
    penalty='l2' #L2 norm is used for penalty for SVM
    svm = LinearSVC(penalty=penalty)
    svm_mne = LinearSVC(penalty=penalty)
    csp = CSP()
    lda = classifier(method='svm')
    mycsp = FBCSP(filter_target='epoched', method='avg_power', bands=None, sum_class=False)
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
    
    
    scores_windows_fbcsp = []
    scores_windows_mne = []
    scores_windows_csp = []
    """
    bpfiltered = np.zeros((raw_bpfiltered.shape[0], epochs_data.shape[0], raw_bpfiltered.shape[2], epochs_data.shape[2]))
    for epoch in range(epochs_data.shape[0]) :
        bpfiltered[:, epoch, :, :] = preproc.bpfilter_data(epochs_data[epoch, :, :])
        
    """
    #pickle.dump(bpfiltered, open('bpfiltered.sav', 'wb'))
    bpfiltered = pickle.load(open('bpfiltered.sav', 'rb'))
    
    for train_idx, test_idx in cv_split:
    
        
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        X_train_csp = mycsp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test_csp = mycsp.transform(epochs_data_train[test_idx])
        
        X_train_mne = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test_mne = csp.transform(epochs_data_train[test_idx])
        # fit classifier
        svm_mne.fit(X_train_mne, y_train)
        svm.fit(X_train_csp, y_train)
        
        lda.fit(epochs_data_train[train_idx], raw_bpfiltered[:, train_idx, :, :], y_train)
    
    
        score_this_window_fbcsp = []
        for n in w_start:
            score_this_window_fbcsp.append(lda.score(epochs_data[test_idx][:, :, n:(n + w_length)], y_test))
        scores_windows_fbcsp.append(score_this_window_fbcsp)
        
        score_this_window_mne = []
        for n in w_start:
            X_test_mne = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
            score_this_window_mne.append(svm_mne.score(X_test_mne, y_test))
        scores_windows_mne.append(score_this_window_mne)
        
        
        score_this_window_csp = []
        for n in w_start:
            X_test_csp = mycsp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
            score_this_window_csp.append(svm.score(X_test_csp, y_test))
        scores_windows_csp.append(score_this_window_csp)



        # Plot scores over time
        w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
         

    
    plt.figure()
    plt.plot(w_times, np.mean(scores_windows_csp, 0), label='Score custom CSP')
    plt.plot(w_times, np.mean(scores_windows_mne, 0), label='Score mne CSP')
    plt.plot(w_times, np.mean(scores_windows_fbcsp, 0), label='Score custom FBCSP')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()
    

    fbpipe = Pipeline([('FBCSP', mycsp), ('SVM', svm)])
    csppipe = Pipeline([('CSP', csp), ('SVM', svm)])
    cl = Pipeline([('FBCSP', classifier(prefiltered=False))])
    
    train_sizesfbcsp, train_scoresfbcsp, test_scoresfbcsp = learning_curve(cl, epochs_data_train, labels, cv=cv, verbose=0)
    train_sizesfb, train_scoresfb, test_scoresfb = learning_curve(fbpipe, epochs_data_train, labels, cv=cv, verbose=0)
    train_sizescsp, train_scorescsp, test_scorescsp = learning_curve(csppipe, epochs_data_train, labels, cv=cv, verbose=0)

    #mean and standard deviation for train scores for FBCSP
    train_scores_meanfbcsp = np.mean(train_scoresfbcsp, axis=1)
    train_scores_stdfbcsp = np.std(train_scoresfbcsp, axis=1)
    
    #mean and standard deviation for train scores for CSP
    train_scores_meanfb = np.mean(train_scoresfb, axis=1)
    train_scores_stdfb = np.std(train_scoresfb, axis=1)
    
    #mean and standard deviation for test scores for MNE CSP
    test_scores_meancsp = np.mean(test_scorescsp, axis=1)
    test_scores_stdcsp = np.std(test_scorescsp, axis=1)
    
    #mean and standard deviation for train scores for MNE CSP
    train_scores_meancsp = np.mean(train_scorescsp, axis=1)
    train_scores_stdcsp = np.std(train_scorescsp, axis=1)
    
    #mean and standard deviation for test scores for CSP
    test_scores_meanfb = np.mean(test_scoresfb, axis=1)
    test_scores_stdfb = np.std(test_scoresfb, axis=1)

    #mean and standard deviation for test scores for FBCSP
    test_scores_meanfbcsp = np.mean(test_scoresfbcsp, axis=1)
    test_scores_stdfbcsp = np.std(test_scoresfbcsp, axis=1)

    #plot learning curve

    # Draw lines
    plt.plot(train_sizesfbcsp, train_scores_meanfbcsp, '--', color="#aa8800",  label="Training score with FBCSP")
    plt.plot(train_sizesfbcsp, test_scores_meanfbcsp, color="#00aa88", label="Cross-validation score with FBCSP")
    
    # Draw bands
    plt.fill_between(train_sizesfbcsp, train_scores_meanfbcsp - train_scores_stdfbcsp, train_scores_meanfbcsp + train_scores_stdfbcsp, color="#DDDDDD")
    plt.fill_between(train_sizesfbcsp, test_scores_meanfbcsp - test_scores_stdfbcsp, test_scores_meanfbcsp + test_scores_stdfbcsp, color="#DDDDDD")

    # Draw lines
    plt.plot(train_sizesfb, train_scores_meanfb, '--', color="#ff0000",  label="Training score with CSP")
    plt.plot(train_sizesfb, test_scores_meanfb, color="#0000ff", label="Cross-validation score with CSP")
    
    # Draw bands
    plt.fill_between(train_sizesfb, train_scores_meanfb - train_scores_stdfb, train_scores_meanfb + train_scores_stdfb, color="#DDDDDD")
    plt.fill_between(train_sizesfb, test_scores_meanfb - test_scores_stdfb, test_scores_meanfb + test_scores_stdfb, color="#DDDDDD")
    
    # Draw lines
    plt.plot(train_sizescsp, train_scores_meancsp, '--', color="#00ffff",  label="Training score with MNE CSP")
    plt.plot(train_sizescsp, test_scores_meancsp, color="#000000", label="Cross-validation score with MNE CSP")
    
    # Draw bands
    plt.fill_between(train_sizescsp, train_scores_meancsp - train_scores_stdcsp, train_scores_meancsp + train_scores_stdcsp, color="#DDDDDD")
    plt.fill_between(train_sizescsp, test_scores_meancsp - test_scores_stdcsp, test_scores_meancsp + test_scores_stdcsp, color="#DDDDDD")
    
    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Cross Validation Accuracy Score"), plt.legend(loc="best")
    #plt.tight_layout()
    plt.show()


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
    #compare_CSP_with_FBCSP(raw) 
    """
    print(score_window(epochs_data_train[3]))
    print("should be")
    print(labels[3])
    """