# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:12:00 2019

@author: Matthaeus Hilpold
"""


from mne.decoding import CSP
import numpy as np
import matplotlib.pyplot as plt
import amplitude_extraction as sp #Melanie Balaz' amplitude_extraction script
from scipy.io import loadmat



#define parameters
EEG_DATA_FILEPATH = "..\\eeg-data\\data_set_IVa_aa_mat\\100Hz\\data_set_IVa_aa.mat" #filepath to the data
CUE_LENGTH = 3.5 #seconds. How long was the cue displayed?
SAMPLING_FREQUENCY = 100 #Hz
CHANNELS = 118 #Number of Channels
BANDS = 5 #Number of frequency bands we are working with



def csp(X_1, X_2, n) :
    """
    :param X_1 : numpy array corresponding to one class. Shape: channels * trials
    :param X_2 : numpy array corresponding to another class. Shape: channels * trials
    :param n : integer number of CSP components to be returned
    """
    eigenvecs_number = int(n/2) #floor to nearest zero
    cov_1 = np.cov(X_1)
    cov_2 = np.cov(X_2)
    eig_value, eig_vector = np.linalg.eigh(np.linalg.inv(cov_2)*cov_1) #eigenvalue decomposition of cov_2^-1 * cov_1 = P D P^-1, where P := matrix of eigen vectors and D := diagonal matrix of eigen values
    return np.concatenate((eig_vector[:, :eigenvecs_number],eig_vector[:, len(eig_vector)-eigenvecs_number:]),axis=1) #return first n/2 and last n/2 components of CSP

def fbcsp(X_1, X_2, n) :
    """
    :param X_1 : numpy array corresponding to eeg data from one class, bandpass filtered. Shape: channels * trials * bands
    :param X_2 : numpy array corresponding to eeg data from another class, bandpass filtered. channels * trials * bands
    :param n : integer number of CSP components to be returned
    """
    fbcsp_result = np.zeros((X_1.shape[0], n, X_1.shape[2]))
    for band in range(X_1.shape[2]) :
        fbcsp_result[:,:,band] = csp(X_1[:,:,band], X_2[:,:,band], n)
    return fbcsp_result

    

if __name__ == '__main__':    
    #read data
    data_set_IVa_aa = loadmat(EEG_DATA_FILEPATH)
    
    raw_data = data_set_IVa_aa['cnt']
    label_pos = data_set_IVa_aa['mrk']['pos'][0][0][0] #cue position (onset) within time signal
    label_class  = data_set_IVa_aa['mrk']['y'][0][0][0] #cue class
    
    #get cues 
    X = np.zeros((CHANNELS, len(label_pos), int(CUE_LENGTH*SAMPLING_FREQUENCY))) #time domain signal
    avg_amp_1 = np.zeros((CHANNELS, (label_class == 1).sum(), BANDS))  #average amplitude (in freq domain) for each band for class label 1
    avg_amp_2 = np.zeros((CHANNELS, (label_class == 2).sum(), BANDS))  #average amplitude (in freq domain) for each band for class label 2
    
    for channel in range(CHANNELS) :                #note to self: this can be optimized both time and memory wise
        for cue in range(np.minimum((label_class == 1).sum(), (label_class == 2).sum())) : #matrix needs to be quadratic. 
            for sample in range(int(SAMPLING_FREQUENCY*CUE_LENGTH)) : 
                X[channel][cue][sample] = raw_data[label_pos[cue]+sample][channel]
            if label_class[cue] == 1 :
                avg_amp_1[channel][cue] = sp.extract_amplitudes(X[channel][cue])
            elif label_class[cue] == 2 :
                avg_amp_2[channel][cue] = sp.extract_amplitudes(X[channel][cue])
            
    fbcsp_result = fbcsp(avg_amp_1, avg_amp_2[:,:80,:], 4)
    
    filtered_result = np.dot(raw_data, fbcsp_result[:, :, 1])
    
    
    #subtracting the mean (mean-centering) can improve CSP
            




