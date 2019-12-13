# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:12:00 2019

@author: Matthaeus Hilpold
"""


import numpy as np
import sys
import matplotlib.pyplot as plt
#import amplitude_extraction_modified as sp #Melanie Balaz' amplitude_extraction script
from scipy.io import loadmat
from scipy.linalg import eigh
from mne.filter import filter_data
from sklearn.preprocessing import scale




class FBCSP :
    
    def __init__(self, features=4, bands=None):  #To keep consistent with signal-processing code.
        self.features = features
        if bands == None :
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
        else:
            self.bands = bands
            
    def fit(self, X, y):
        
        X_1, X_2 = self.separate_classes(X, y)
        self.filters_ = self.fbcsp(X_1, X_2, self.features)
        
        return self
    
    def transform(self, X, band=1) :
        
        return np.dot(X, self.filters_[:, :, band].T)
    
    def fit_transform(self,raw, X, y, band=1) :
        
        
        self.fit(X, y)
        return self.transform(raw, band)
        
        
    
    def csp(self, X_1, X_2, n=4) :
        """
        :param X_1 : numpy array corresponding to one class. Shape: channels * trials * samples
        :param X_2 : numpy array corresponding to another class. Shape: channels * trials *samples
        :param n : integer number of CSP components to be returned
        :param extraction : String method used for extraction of CSP features. If 'simple' then argmax(|log(eig)|) is used. Default: 'simple' 
        """
        #eigenvecs_number = int(n/2) #floor to nearest zero
        
        #average over trials per class.
        avg_1 = np.mean(X_1, axis=0)
        avg_2 = np.mean(X_2, axis=0)
        
        # compute covariance matrices.
        cov_1 = np.cov(avg_1)
        cov_2 = np.cov(avg_2)
    
    
        eig_value, eig_vector = eigh(cov_2, cov_1)
        
        #find the eigen values that account for the highest separability among the two classes
        feature_ind = np.argpartition(np.abs(np.log10(eig_value)), -n)[-4:] #is argpartition := argmax even necessary? Eigh already returns a sorted eig_value vector, so picking the first n/2 and the last n/2 would already suffice?
        
        #pick and return CSP features from eigenvectors
        return np.array([eig_vector[idx] for idx in feature_ind])
    
        
        #note to self: https://www.ncbi.nlm.nih.gov/pubmed/18632362 if it will be expanded to multiclass. 
        """
        #v_b_i extracted according to: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3314883/
        if extraction != 'simple' :
            for cue in range(X_1.shape[0]) :
                for channel in range(X_1.shape[1]) :
                    Wt_E_Et_W = np.dot(np.dot(np.dot(np.transpose(result), X_1), np.transpose(X_1)), result)
                    diag_Wt_E_Et_W = np.diag(Wt_E_Et_W)
                    sum_diag_Wt_E_Et_W= np.sum(diag_Wt_E_Et_W)
                    log_v_bi = np.log(np.divide(diag_Wt_E_Et_W, sum_diag_Wt_E_Et_W))
                    return log_v_bi
        """       
    
    
    def fbcsp(self, X_1, X_2, n) :
        """
        :param X_1 : numpy array corresponding to eeg data from one class, bandpass filtered. Shape: trials * channels * bands * samples
        :param X_2 : numpy array corresponding to eeg data from another class, bandpass filtered. trials * channels * bands * samples
        :param n : integer number of CSP components to be returned
        :param trialwise : Boolean  apply CSP single-trial. Default: False
        :param extraction : String extraction method. Default: 'simple'
        """
        
        fbcsp_result = np.zeros((n, X_1.shape[1], X_1.shape[2]))
        
        for band in range(X_1.shape[2]) :
            fbcsp_result[:,:,band] = self.csp(X_1[:,:,band,:], X_2[:,:,band,:], n)
            
        return fbcsp_result
    
    def separate_classes(self, X, y) :
        
        max_number_trials = np.minimum((y == 1).sum(), (y == 2).sum())
        
        class_1 = np.array([X[trial, :, :, :] for trial in range(max_number_trials) if y[trial] == 1. ])
        class_2 = np.array([X[trial, :, :, :] for trial in range(max_number_trials) if y[trial] == 2.])
    
        return class_1, class_2

    
    def preproc(self, raw_data, label_pos, Label_class, fs, windowlength) :
        
        raw_data = raw_data.astype(float) #filter_data requires float64 unfortunately, therefore we must cast all int16 to f64
        
        #center data. remove mean from each channel.
        raw_data_centered = np.array([(raw_data.T[channel] - np.mean(raw_data.T[channel]))  for channel in range(raw_data.shape[1])]) # maybe sklearn.preprocessing.scale better?        
        
        
        #is centering the data really necessary for FBCSP?
        #according to: https://hal.inria.fr/hal-01182728/document the mean should already be 0.
        #cit. : 'Notably, such a computation ofcovariance matrices assumes the EEG signals to have a zeromean (which is true in practice for band-pass filtered signals).'
        
        
        #band-pass filter raw data
        raw_data_filtered = np.zeros((raw_data_centered.shape[0], len(self.bands.items()), raw_data_centered.shape[1]))
        
        for channel in range(raw_data_centered.shape[0]) :
            i = 0
            for name, band in self.bands.items() :
                raw_data_filtered[channel][i] = filter_data(raw_data_centered[channel], fs,  band[0], band[1], copy=True)
                i += 1
        
        
        #extract trials 
        max_number_trials = np.minimum((label_class == 1).sum(), (label_class == 2).sum()) # matrices need to be quadratic for csp
        samples_per_trial = int(fs*windowlength)     #number of samples in a window
        X = np.zeros((len(label_pos), raw_data_filtered.shape[0], len(self.bands.items()), samples_per_trial)) #filter-bank time domain signal with trials
        
        #trials of each class extracted from the time domain signal    
        class_1 = []
        class_2 = []
        
        #class_1 = np.zeros(((label_class == 1).sum(), CHANNELS, number_cue_samples))
        #class_2 = np.zeros(((label_class == 2).sum(), CHANNELS, number_cue_samples))
        
        for trial in range(max_number_trials) :
            X[trial] = raw_data_filtered[:,:,label_pos[trial]:(label_pos[trial]+samples_per_trial)]
            if label_class[trial] == 1. :
                class_1.append(raw_data_filtered[:,:,label_pos[trial]:(label_pos[trial]+samples_per_trial)])
            if label_class[trial] == 2. :
                class_2.append(raw_data_filtered[:,:,label_pos[trial]:(label_pos[trial]+samples_per_trial)])
                
        class_1 = np.array(class_1)
        class_2 = np.array(class_2)
        
        return X, class_1, class_2
    

if __name__ == '__main__':   
    #define parameters
    EEG_DATA_FILEPATH = "..\\..\\..\\eeg-data\\data_set_IVa_aa_mat\\100Hz\\data_set_IVa_aa.mat" #filepath to the data
    #window_length = float(sys.argv[1]) #seconds. How long was the cue displayed? 
    #fs = int(sys.argv[2]) #Hz
    
    window_length = 3.5 #seconds. How long was the cue displayed? 
    fs = 100 #Hz
    
    #read data
    data_set_IVa_aa = loadmat(EEG_DATA_FILEPATH)

    raw_data = data_set_IVa_aa['cnt']
    label_pos = data_set_IVa_aa['mrk']['pos'][0][0][0] #trial position (onset) within time signal
    label_class  = data_set_IVa_aa['mrk']['y'][0][0][0] #trial class
    
    fbcsp = FBCSP()
    X, class_1, class_2 = fbcsp.preproc(raw_data, label_pos, label_class, fs, window_length)
    fbcsp.fit_transform(raw_data, X, label_class)
    fbcsp.fit(X, label_class)
    print(fbcsp.filters_.shape)
    fbcsp.transform(raw_data)
    print(raw_data.shape)
    
    #fbcsp_result_regularized = fbcsp(avg_amp_1[:,:max_number_cues,:], avg_amp_2[:,:max_number_cues,:], 4, regularization=True)
    
    