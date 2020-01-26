# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:10:46 2019

@author: Matthaeus Hilpold
"""

import numpy as np
from mne.filter import filter_data

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

