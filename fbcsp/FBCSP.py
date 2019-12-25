# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:12:00 2019

@author: Matthaeus Hilpold
"""


import numpy as np
#import sys
#import matplotlib.pyplot as plt
import amplitude_extraction as sp #Melanie Balaz' amplitude_extraction script
from scipy.io import loadmat
from scipy.linalg import eigh
from mne.filter import filter_data
#from sklearn.preprocessing import scale



class FBCSP :
    
    def __init__(self, components=4, bands=None, filter_target='epoched', classes=2, concatenate=False, avg_band=False):
        """
        :param components : int. number of CSP components the filter will contain
        :param bands : dict. one or multiple frequency bands
        :param filter_targets : str. If 'epoched' then transform will apply the filter to the epoched data, if 'raw' transform will be applied to the raw data.
        :param classes : int. number of classes that will be evaluated. Default 2 for binary classification. Multiclass will be evaluated as one-versus-rest.
        :param concatenate : bool. If true, epoched data will be flattened to shape channels*samples when transforming
        
        :attribute filters_ : numpy array. CSP filter.  Shape: channels * samples
        """
        self.avg_band = avg_band
        self.concatenate = concatenate
        self.components  = components
        self.filter_target = filter_target
        self.classes = classes
        if bands == None :
            self.bands = {'sub-band' : (7, 30)}
        elif bands == 'all' :
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
        else:
            self.bands = bands
            
    def fit(self, X, y):
        """
        :param X : 4d or 3d numpy array of epoched data. Shape: band (optional) * epochs * channels * samples
        :param y : 1d numpy array of class labels
        """
        
        if X.ndim == 3 :
            X_1, X_2 = self.separate_classes(X, y)
            self.filters_ = self.csp(X_1, X_2, self.components)
            print('CSP filters fit. Filter shape: ' + str(self.filters_.shape))
        else :
            X_1, X_2 = self.separate_classes(X, y)
            self.filters_ = self.fbcsp(X_1, X_2, self.components)
            print('CSP filters fit. Filter shape: ' + str(self.filters_.shape))
        
        return self
    
    def transform(self, X) :
        """
        :param X : 4d numpy array time-domain data. Shape: band (optional) * epochs (optional) * channels * samples
        :param y : 1d numpy array of class labels
        """
        
        if self.filter_target == 'epoched' and len(self.bands) == 1:
            filtered_data = [] 
            for epoch in range(X.shape[0]) :
                filtered_data.append(np.dot(self.filters_.T, X[epoch, :, :]))     
            filtered_data = np.array(filtered_data)
            if self.concatenate == True :
                filtered_data = self.concatenate_result(filtered_data)
                print('Single-band epoched data filtered and concatenated. Shape: ' + str(filtered_data.shape))
                return filtered_data
            elif self.avg_band == True:
                filtered_data = self.average_band_amplitude(filtered_data)
                print('Average band amplitude of single-band epoched data filtered. Shape: ' + str(filtered_data.shape))
                return filtered_data
            else :
                print('Single-band epoched data filtered. Shape: ' + str(filtered_data.shape))
                return filtered_data
         
        elif self.filter_target == 'epoched' and len(self.bands) != 1:
            filtered_data = []
            for band in range(len(self.bands)) :
                for epoch in range(X.shape[1]) :
                    filtered_data.append(np.dot(self.filters_[band, :, :].T, X[band, epoch, :, :]))
            print('Epoched data filtered and filter-banked. Shape: ' + str(filtered_data.shape))
            return filter_data
        
        elif self.filter_target == 'raw' and len(self.bands) == 1:
            filtered_data = np.dot(self.filters_.T, X)
            print('Single-band raw data filtered. Shape: ' + str(filtered_data.shape))
            return filtered_data

        elif self.filter_target == 'raw' and len(self.bands) != 1:
            #TODO
            print('TODO')
            
    def fit_transform(self, X, y) :
        
        
        self.fit(X, y)
        return self.transform(X)
        
        
    
    def csp(self, X_1, X_2, k=4) :
        """
        :param X_1 : numpy array corresponding to one class. Shape: epochs * channels * samples
        :param X_2 : numpy array corresponding to another class. Shape: epochs * channels * samples
        :param k : integer number of CSP components to be returned
        """
        
        # compute covariance matrices per epoch.
        cov_1 = np.array([np.cov(X_1[epoch]) for epoch in range(X_1.shape[0])])
        cov_2 = np.array([np.cov(X_2[epoch]) for epoch in range(X_2.shape[0])])
        
        #average covariance matrices over epochs.
        avg_1 = np.mean(cov_1, axis=0)
        avg_2 = np.mean(cov_2, axis=0)
        
        #compute eigenvalues and eigenvectors
        eig_value, eig_vector = eigh(avg_1, avg_2)
        
        #find the eigen values that account for the highest separability among the two classes
        component_ind = np.flip(np.argsort(np.abs(np.log10(eig_value)))) #get the indices of the most incisive eigenvalues
        component_ind = component_ind[:k] #select first k components
        
        #pick and return CSP components from eigenvectors
        #import pdb
        #pdb.set_trace()
        return np.array([eig_vector[:, idx] for idx in component_ind]).T
    
        
        #note to self: https://www.ncbi.nlm.nih.gov/pubmed/18632362 if it will be expanded to multiclass. 
    
    
    def fbcsp(self, X_1, X_2, k) :
        """
        :param X_1 : numpy array corresponding to eeg data from one class, bandpass filtered. Shape: bands *epochs * channels * samples
        :param X_2 : numpy array corresponding to eeg data from another class, bandpass filtered. Shape: bands* epochs * channels * samples
        :param k : integer number of CSP components to be returned
        """
        
        fbcsp_result = np.zeros((X_1.shape[0], k, X_1.shape[3]))
        
        #apply CSP to each band
        for band in range(X_1.shape[2]) :
            fbcsp_result[band,:,:] = self.csp(X_1[band,:,:,:], X_2[band,:,:,:], k)
            
        return fbcsp_result
    
    def concatenate_result(self, X) :
        """
        :param X : numpy array of shape epochs * channels * samples
        """
        X = X.transpose(2,0,1)
        X = X.reshape(X.shape[2], -1)
        
        return X
    
    def average_band_amplitude(self, X) :
        """
        :param X : numpy array of epoched eeg data. Shape: epochs * channels * samples
        """
        bandavg = np.zeros((X.shape[0], 4, X.shape[1]))
        
        for epoch in range(X.shape[0]) :
            bandavg[epoch] = (sp.extract_amplitudes(X[epoch,:,:]))
        
        X = bandavg.reshape(bandavg.shape[0], -1)
        return X
    
        
    
    def separate_classes(self, X, y) :
        
        """
        :param X : numpy array of epoched eeg data. Shape: bands * epochs * channels * samples
        :param y : numpy array of class labels
        """
        
        if len(self.bands) == 1 :
            #Binary case
            if self.classes == 2 :
                #extract class labels
                classarr = np.unique(y)
                
                """"
                #ensure that both classes have the same amount of trials.
                max_number_epochs = np.minimum((y == classarr[0]).sum(), (y == classarr[1]).sum())
                """
                
                #extract classes
                class_1 = np.array([X[epoch, :, :] for epoch in range((y == classarr[0]).sum()) if y[epoch] == classarr[0]])
                class_2 = np.array([X[epoch, :, :] for epoch in range((y == classarr[1]).sum()) if y[epoch] == classarr[1]])
            
                return class_1, class_2
            
            #Multi-class case
            else :
                #TODO
                print('TODO')
        else :
            #Binary case
            if self.classes == 2 :
                #extract class labels
                classarr = np.unique(y)
                
                #extract classes
                class_1 = np.array([X[:, epoch, :, :] for epoch in range((y == classarr[0]).sum()) if y[epoch] == classarr[0]])
                class_2 = np.array([X[:, epoch, :, :] for epoch in range((y == classarr[1]).sum()) if y[epoch] == classarr[1]])
            
                return class_1, class_2
            
            #Multi-class case
            else :
                #TODO
                print('TODO')

class Preproc :
    
    def __init__(self, bands=None, classes=2, fs=100, windowlength=3.5):
        """
        :param bands : dict. one or multiple frequency bands
        :param classes : int. number of classes that will be evaluated. Default 2 for binary classification. Multiclass will be evaluated as one-versus-rest.
        :param fs : int. sampling frequency
        :param windowlength : float. length of a window in seconds. 
        
        :attribute epoched_data_ : numpy array of epoched data. Shape: bands * epochs *channels * samples
        """

        self.classes = classes
        self.fs = fs
        self.windowlength = windowlength
        
        if bands == None :
            self.bands = {'sub-band' : (7, 30)}
        elif bands == 'all' :
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
        else:
            self.bands = bands
    
    def center_data(self, raw_data) :
        
        #center data. remove mean from each channel.
        raw_data = np.array([(raw_data.T[channel] - np.mean(raw_data.T[channel]))  for channel in range(raw_data.shape[0])])
        print("Centering complete. Centered data shape: " + str(raw_data.shape))

    def bpfilter_data(self, raw_data) :
        
        if len(self.bands.items()) > 1 :
            #band-pass filter raw data
            raw_data_filtered = np.zeros((len(self.bands.items()), raw_data.shape[0], raw_data.shape[1]))
        
            for channel in range(raw_data.shape[0]) :
                i = 0
                for name, band in self.bands.items() :
                    raw_data_filtered[i, channel, :] = filter_data(raw_data[channel], self.fs,  band[0], band[1], copy=True,verbose=0)
                    i += 1
            raw_data = raw_data_filtered
            print("Bandpass filtering complete. Bandpass filtered data shape: " + str(raw_data.shape))
        
        else :
            for key, value in self.bands.items() :
                raw_data = np.array([filter_data(raw_data[channel], self.fs,  value[0], value[1], copy=True, verbose=0) for channel in range(raw_data.shape[0])])
            print("Bandpass filtering complete. Bandpass filtered data shape: " + str(raw_data.shape))
     
    def epoch(self, raw_data, label_pos) :
        
        #extract epochs         
        samples_per_epoch = int(self.fs*self.windowlength)     #number of samples in a window
        
        if len(self.bands) > 1 :
            
            self.epoched_data_ = np.zeros((len(self.bands.items()), len(label_pos), raw_data.shape[0], samples_per_epoch)) #filter-bank time domain signal with epochs
            
            for epoch in range(len(label_pos)) :
                self.epoched_data_[:, epoch, :, :] = raw_data[:,:,label_pos[epoch]:(label_pos[epoch]+samples_per_epoch)]
            print("Epoching complete. Epoched data shape: " + str(self.epoched_data_.shape))
            
        else :
            self.epoched_data_ = np.zeros((len(label_pos), raw_data.shape[0], samples_per_epoch)) #time domain signal with epochs for a single band.
            
            for epoch in range(len(label_pos)) :
                self.epoched_data_[epoch, :, :] = raw_data[:,label_pos[epoch]:(label_pos[epoch]+samples_per_epoch)]
            print("Epoching complete. Epoched data shape: " + str(self.epoched_data_.shape))
                
    def preproc(self, raw_data, label_pos) :
        
        raw_data = raw_data.astype(float) #filter_data requires float64 unfortunately, therefore int16 needs to be cast to f64
        
        #center data. remove mean from each channel.
        self.center_data(raw_data) # maybe sklearn.preprocessing.scale better?        
        
        #band-pass filter raw data
        self.bpfilter_data(raw_data)
               
        #extract epochs         
        self.epoch(raw_data, label_pos)
        
        print("Pre-processing complete. Data shape: " + str(self.epoched_data_.shape))
        
        return self.epoched_data_

if __name__ == '__main__':   
    #define parameters
    EEG_DATA_FILEPATH = "..\\..\\..\\data_set_IVa_aa_mat\\100Hz\\data_set_IVa_aa.mat" #filepath to the data
    #window_length = float(sys.argv[1]) #seconds. How long was the cue displayed? 
    #fs = int(sys.argv[2]) #Hz
    
    window_length = 3.5 #seconds. How long was the cue displayed? 
    fs = 100 #Hz
    
    #read data
    data_set_IVa_aa = loadmat(EEG_DATA_FILEPATH)

    raw_data = data_set_IVa_aa['cnt'].T
    label_pos = data_set_IVa_aa['mrk']['pos'][0][0][0] #epoch position (onset) within time signal
    label_class  = data_set_IVa_aa['mrk']['y'][0][0][0] #epoch class
    
    p = Preproc(bands=None, fs=fs, windowlength=window_length)
    
    X = p.preproc(raw_data, label_pos)
    
    csp = FBCSP(filter_target='raw')
    print('-----------------------------------------------------')
    csp.fit(X, label_class)
    result_1 = csp.transform(raw_data)
    print('-----------------------------------------------------')
    csp2 = FBCSP(filter_target='epoched', concatenate = True)
    result_2 = csp2.fit_transform(X, label_class)
    print('-----------------------------------------------------')
    csp3 = FBCSP(filter_target='epoched', concatenate = False)
    result_3 = csp3.fit_transform(X, label_class)
    print('-----------------------------------------------------')
    csp4 = FBCSP(filter_target='epoched', avg_band=True)
    result_4 = csp3.fit_transform(X, label_class)