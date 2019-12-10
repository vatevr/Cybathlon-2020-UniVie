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



def csp(X_1, X_2, n, regularization=False) :
    """
    :param X_1 : numpy array corresponding to one class. Shape: channels * trials
    :param X_2 : numpy array corresponding to another class. Shape: channels * trials
    :param n : integer number of CSP components to be returned
    """
    eigenvecs_number = int(n/2) #floor to nearest zero
    cov_1 = np.cov(X_1)
    cov_2 = np.cov(X_2)
    eig_value, eig_vector = np.linalg.eigh(np.linalg.inv(cov_2)*cov_1) #eigenvalue decomposition of cov_2^-1 * cov_1 = P D P^-1, where P := matrix of eigen vectors and D := diagonal matrix of eigen values
    
    first_eigenvecs = eig_vector[:, :eigenvecs_number]
    last_eigenvecs = eig_vector[:, len(eig_vector)-eigenvecs_number:]
    result = np.concatenate((last_eigenvecs, first_eigenvecs),axis=1)
    
    #Regularized according to: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3314883/
    if regularization == True :
        for cue in range(X_1.shape[0]) :
            for channel in range(X_1.shape[1]) : 
                regularized_result = np.concatenate(
                                         np.divide(                              #for each cue in class 1
                                            np.diag( 
                                                np.log(
                                                    np.dot(
                                                        np.dot(
                                                                np.dot(np.transpose(result), X_1)
                                                        , np.transpose(X_1)
                                                        )
                                                    , result
                                                    )
                                                )
                                            )
                                            , np.sum(
                                                np.diag(
                                                    np.dot(
                                                        np.dot(
                                                                np.dot(np.transpose(result), X_1)
                                                        , np.transpose(X_1)
                                                        )
                                                    , result
                                                    )   
                                                )
                                            )
                                        ), 
                                        np.divide(                              #for each cue in class 2
                                            np.diag( 
                                                np.log(
                                                    np.dot(
                                                        np.dot(
                                                                np.dot(np.transpose(result), X_2)
                                                        , np.transpose(X_2)
                                                        )
                                                    , result
                                                    )
                                                )
                                            )
                                            , np.sum(
                                                np.diag(
                                                    np.dot(
                                                        np.dot(
                                                                np.dot(np.transpose(result), X_2)
                                                        , np.transpose(X_2)
                                                        )
                                                    , result
                                                    )   
                                                )
                                            )
                                        )
                                    )
            
        return regularized_result
    return result #return first n/2 and last n/2 components of CSP

def fbcsp(X_1, X_2, n) :
    """
    :param X_1 : numpy array corresponding to eeg data from one class, bandpass filtered. Shape: channels * trials * bands
    :param X_2 : numpy array corresponding to eeg data from another class, bandpass filtered. channels * trials * bands
    :param n : integer number of CSP components to be returned
    """
    print(X_1)
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
    max_number_cues = np.minimum((label_class == 1).sum(), (label_class == 2).sum()) # matrices need to be quadratic for csp
    number_cue_samples = int(SAMPLING_FREQUENCY*CUE_LENGTH)
    X = np.zeros((len(label_pos), number_cue_samples, CHANNELS)) #time domain signal
    class_1 = np.zeros(((max_number_cues, CHANNELS, number_cue_samples)))   #average amplitude (in freq domain) for each band for class label 1
    class_2 = np.zeros(((max_number_cues, CHANNELS, number_cue_samples)))  #average amplitude (in freq domain) for each band for class label 2
    
    cnt = 0
    for channel in range(CHANNELS) :                #note to self: this can be optimized both time and memory wise   
        for sample in range(number_cue_samples) :
            i = 0
            j = 0
            for cue in range(max_number_cues) :
                cnt += 1
                X[cue][sample][channel] = raw_data[label_pos[cue]+sample][channel]
                if label_class[cue] == 1. :
                    class_1[i][channel][sample] = raw_data[label_pos[cue]+sample][channel]  #need to have samples last, otherwise it won't work together with Melanies code
                    i += 1
                if label_class[cue] == 2. :
                    class_2[j][channel][sample] = raw_data[label_pos[cue]+sample][channel]
                    j += 1
        
    
                    
    avg_amp_1 = np.array(sp.extract_amplitudes(class_1))    #need to convert to numpy array, because it returns a list
    avg_amp_2 = np.array(sp.extract_amplitudes(class_2))
    
    fbcsp_result = fbcsp(avg_amp_1, avg_amp_2, 4)
    
    filtered_result = np.dot(raw_data, fbcsp_result[:, :, 1])
    
    
    #subtracting the mean (mean-centering) can improve CSP
            




