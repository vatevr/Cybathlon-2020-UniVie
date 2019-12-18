# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:06:56 2019

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



class FBCSP (OVBox):
    
    def __init__(self, components=4, bands=None, filter_target='epoched', classes=2, concatenate=False, avg_band=False):
        """
        :param components : int. number of CSP components the filter will contain
        :param bands : dict. one or multiple frequency bands
        :param filter_targets : str. If 'epoched' then transform will apply the filter to the epoched data, if 'raw' transform will be applied to the raw data.
        :param classes : int. number of classes that will be evaluated. Default 2 for binary classification. Multiclass will be evaluated as one-versus-rest.
        :param concatenate : bool. If true, epoched data will be flattened to shape channels*samples when transforming
        
        :attribute filters_ : numpy array. CSP filter.  Shape: channels * samples
        """
        OVBox.__init__(self)
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
    
    def initialize(self) :
        #TODO
        print('TODO')
    
    def process(self) :
        #TODO
        print('TODO')

    
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
box = FBCSP()
