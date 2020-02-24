import numpy as np
#import sys
#import matplotlib.pyplot as plt
import amplitude_extraction as sp #Melanie Balaz' amplitude_extraction script
from scipy.io import loadmat
from scipy.linalg import eigh
from mne.filter import filter_data
#from mne import cov as mne_cov
#from feature_selector import feature_selector
from scipy import linalg

#from sklearn.preprocessing import scale



class FBCSP :
    
    def __init__(self, components=4, bands='all', filter_target='epoched', classes=2, method='avg_power', sum_class=False):
        """
        :param components : int. number of CSP components the filter will contain
        :param bands : dict. one or multiple frequency bands
        :param filter_targets : str. If 'epoched' then transform will apply the filter to the epoched data, if 'raw' transform will be applied to the raw data.
        :param classes : int. number of classes that will be evaluated. Default 2 for binary classification. Multiclass will be evaluated as one-versus-rest.
        :param avg_band : If True average band amplitude of the computed csp filters will be computed.
        
        :attribute filters_ : numpy array. CSP filter.  Shape: channels * samples
        """
        self.method = method
        self.components  = components
        self.filter_target = filter_target
        self.classes = classes
        self.sum_class = sum_class
        if bands == None :
            self.bands = {'sub-band' : (7, 30)}
        elif bands == 'all' :
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta-1': (12, 20), 'beta-2': (20,30), 'gamma': (30, 45)}
        else:
            self.bands = bands
            
    def fit(self, X, y):
        """
        :param X : 4d or 3d numpy array of epoched data. Shape: band (optional) * epochs * channels * samples
        :param y : 1d numpy array of class labels
        """
        
        if X.ndim == 3 :
            X_1, X_2 = self.separate_classes(X, y)
            if self.sum_class == True :
                self.filters_, self.patterns_ = self.csp_sum_class(X_1, X_2, self.components)
            else :
                self.filters_, self.patterns_ = self.csp(X_1, X_2, self.components)
            print('CSP filters fit. Filter shape: ' + str(self.filters_.shape))
        else :
            X_1, X_2 = self.separate_classes(X, y)
            self.filters_, self.patterns_ = self.fbcsp(X_1, X_2, self.components)
            print('FBCSP filters fit. Filter shape: ' + str(self.filters_.shape))
        
        return self
    
    def transform(self, X) :
        """
        :param X : 4d numpy array time-domain data. Shape: band (optional) * epochs (optional) * channels * samples
        """
        
        if self.filter_target == 'epoched' and X.ndim == 3 :
            filtered_data = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
            
            if self.method == 'avg_power' :
                filtered_data = self.average_band_power(filtered_data)
                print('Average band power of single-band epoched data filtered. Shape: ' + str(filtered_data.shape))

            elif self.method == 'avg_amplitude':
                filtered_data = self.average_band_amplitude(filtered_data)
                print('Average band amplitude of single-band epoched data filtered. Shape: ' + str(filtered_data.shape))

            else :
                print('Single-band epoched data filtered. Shape: ' + str(filtered_data.shape))
            return filtered_data
         
        elif self.filter_target == 'epoched' and X.ndim > 3:
            filtered_data = np.zeros((X.shape[0], X.shape[1], self.components, X.shape[3]))
            for band in range(X.shape[0]) :
                for epoch in range(X.shape[1]) :
                    filtered_data[band, epoch, :] = np.dot(self.filters_[band] , X[band, epoch, :, :])
            filtered_data = np.array(filtered_data)
            
            if self.method == 'avg_power' :
                filtered_data = self.average_band_power(filtered_data)
                print('Average band power of filter-bank epoched data filtered. Shape: ' + str(filtered_data.shape))

            else :
                print('Epoched data filtered and filter-banked. Shape: ' + str(filtered_data.shape))
            return filtered_data
        
        elif self.filter_target == 'raw' and len(self.bands) == 1:
            filtered_data = np.dot(self.filters_, X)
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
        #import pdb
        #pdb.set_trace()
        
        # compute covariance matrices per epoch.
        cov_1 = np.array([np.cov(X_1[epoch]) 
                          for epoch in range(X_1.shape[0])])
        cov_2 = np.array([np.cov(X_2[epoch])
                          for epoch in range(X_2.shape[0])])
        """
        cov_1 = np.array([mne_cov._regularized_covariance(X_1[epoch], reg=None,method_params=None, rank=None) 
                          for epoch in range(X_1.shape[0])])
        cov_2 = np.array([mne_cov._regularized_covariance(X_2[epoch], reg=None,method_params=None, rank=None) 
                          for epoch in range(X_2.shape[0])])
       """
        #average covariance matrices over epochs.
        avg_1 = np.mean(cov_1, axis=0)
        avg_2 = np.mean(cov_2, axis=0)
        
        
        #compute eigenvalues and eigenvectors
        eig_value, eig_vector = eigh(avg_1, avg_2)
        
        #find the eigen values that account for the highest separability among the two classes
        component_ind = np.flip(np.argsort(np.abs(np.log10(eig_value)))) #get the indices of the most incisive eigenvalues
        eig_vector = eig_vector[:, component_ind] #sort eigenvector matrix
        component_ind = component_ind[:k] #select first k components
        
        
        patterns = linalg.pinv2(eig_vector) #copied from mne library
        
        #pick and return CSP components from eigenvectors
        return np.array([eig_vector[:, idx] for idx in component_ind]), patterns
    
        
        #note to self: https://www.ncbi.nlm.nih.gov/pubmed/18632362 if it will be expanded to multiclass. 
        
    def csp_sum_class(self, X_1, X_2, k=4) :
        """
        :param X_1 : numpy array corresponding to one class. Shape: epochs * channels * samples
        :param X_2 : numpy array corresponding to another class. Shape: epochs * channels * samples
        :param k : integer number of CSP components to be returned
        
        Alternative to computing the eigenvalue decomposition of class 1 vs 
        class 2 is computing eigenvalue decomposition of 
        class 1 vs (class 1 + class 2). 
        Some sources, including the one i should be basing my code on, don't
        seem to be doing it that way, but several other sources do.
        E.g.: Kai Keng Ang et al. , Front. Neurosci., 29 March 2012.
        """
        
        # compute covariance matrices per epoch. Mne covariance gives values 
        #that are different by several factors, but numpy seems to give better 
        #accuracies
        
        
        cov_1 = np.array([np.cov(X_1[epoch]) 
                          for epoch in range(X_1.shape[0])])
        cov_2 = np.array([np.cov(X_2[epoch])
                          for epoch in range(X_2.shape[0])])
        """
        
        cov_1 = np.array([mne_cov._regularized_covariance(X_1[epoch], reg=None,method_params=None, rank=None) 
                          for epoch in range(X_1.shape[0])])
        cov_2 = np.array([mne_cov._regularized_covariance(X_2[epoch], reg=None,method_params=None, rank=None) 
                          for epoch in range(X_2.shape[0])])
        """
        #average covariance matrices over epochs.
        avg_1 = np.mean(cov_1, axis=0)
        avg_2 = np.mean(cov_2, axis=0)
        
        
        #compute eigenvalues and eigenvectors
        eig_value, eig_vector = eigh(avg_1, (avg_1 + avg_2))
        
        #find the eigen values that account for the highest separability among the two classes
        component_ind = np.flip(np.argsort(np.abs(eig_value-0.5))) #get the indices of the most incisive eigenvalues
        eig_vector = eig_vector[:, component_ind] #sort eigenvector matrix
        component_ind = component_ind[:k] #select first k components
        
        patterns = linalg.pinv2(eig_vector)                 #shamelessly copied from mne
        #pick and return CSP components from eigenvectors
        #import pdb
        #pdb.set_trace()
        return np.array([eig_vector[:, idx] for idx in component_ind]), patterns
    
        
        #note to self: https://www.ncbi.nlm.nih.gov/pubmed/18632362 if it will be expanded to multiclass. 
    
    
    def fbcsp(self, X_1, X_2, k) :
        """
        :param X_1 : numpy array corresponding to eeg data from one class, bandpass filtered. Shape: bands *epochs * channels * samples
        :param X_2 : numpy array corresponding to eeg data from another class, bandpass filtered. Shape: bands* epochs * channels * samples
        :param k : integer number of CSP components to be returned
        """
        #import pdb
        #pdb.set_trace()
        
        fbcsp_result = np.zeros((X_1.shape[0], k, X_1.shape[2]))
        fbcsp_patterns = np.zeros((X_1.shape[0], X_1.shape[2], X_1.shape[2]))
        
        #apply CSP to each band
        for band in range(X_1.shape[0]) :
            fbcsp_result[band,:,:], fbcsp_patterns[band, :, :] = self.csp(X_1[band,:,:,:], X_2[band,:,:,:], k)
        if self.sum_class == True :
            fbcsp_result[band,:,:], fbcsp_patterns[band,:,:] = self.csp_sum_class(X_1[band,:,:,:], X_2[band,:,:,:], k) 
            
        return fbcsp_result, fbcsp_patterns
    
    def concatenate_result(self, X) :
        """
        :param X : numpy array of shape bands *  epochs * components
        """
        X = X.transpose(2,0,1)
        X = X.reshape(X.shape[2], -1)
        
        return X
    
    def average_band_amplitude(self, X) :
        """
        :param X : numpy array of epoched eeg data. Shape: epochs * channels/components * samples
        """
        #import pdb
        #pdb.set_trace()
        bandavg = np.zeros((5, X.shape[0], X.shape[1]))
        for epoch in range(X.shape[0]) :
            bandavg[:,epoch,:] = sp.extract_amplitudes(X[epoch,:,:])

        concat = np.zeros((bandavg.shape[1], bandavg.shape[0]*bandavg.shape[2]))
        for band in range(bandavg.shape[0]) :
            for epoch in range(bandavg.shape[1]) :
                for component in range(bandavg.shape[2]) :
                    concat[epoch, (band*component)+component] = (bandavg[band, epoch, component] ** 2) #concatenate square amplitude per band
        
        return np.asarray(concat)
    
    def average_band_power(self, X) :
        """
        :param X : numpy array of epoched eeg data. Shape: bands (optional) * epochs * channels/components * samples
        """
        if X.ndim == 3 :
            avg_power = np.zeros((X.shape[0], X.shape[1]))
            avg_power = np.mean(np.power(X, 2), axis=2)
            avg_power = np.log(avg_power)
            return avg_power
        else :
            avg_band_power = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
            for band in range(X.shape[0]) :
                avg_band_power[band] = (X[band] ** 2).mean(axis=2)
            avg_band_power = np.log(avg_band_power)
            return avg_band_power
        
    
    def separate_classes(self, X, y) :
        
        """
        :param X : numpy array of epoched eeg data. Shape: bands * epochs * channels * samples
        :param y : numpy array of class labels
        """
        
        if X.ndim == 3 :
            #Binary case
            if self.classes == 2 :
                #extract class labels
                classarr = np.unique(y)
                
                """"
                #ensure that both classes have the same amount of trials.
                max_number_epochs = np.minimum((y == classarr[0]).sum(), (y == classarr[1]).sum())
                """
                
                #extract classes
                class_1 = np.array([X[epoch, :, :] for epoch in range(len(y)) if y[epoch] == classarr[0]])
                class_2 = np.array([X[epoch, :, :] for epoch in range(len(y)) if y[epoch] == classarr[1]])
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
                class_1 = np.array([X[:, epoch, :, :] for epoch in range(len(y)) if y[epoch] == classarr[0]])
                class_2 = np.array([X[:, epoch, :, :] for epoch in range(len(y)) if y[epoch] == classarr[1]])
                
                class_1 = np.transpose(class_1, (1, 0, 2, 3))
                class_2 = np.transpose(class_2, (1, 0, 2, 3))
                
                return class_1, class_2
            
            #Multi-class case
            else :
                #TODO
                print('TODO')
                
    def plot_patterns(self, info, layout=None, title='MNE CSP 7-30 eegbci') : #shamelessly adapted from mne
        from mne import EvokedArray
        import copy as cp
        #import pdb
        #pdb.set_trace()
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        
        
        if len(self.bands) < 2 :
            patterns = EvokedArray(self.patterns_.T, info, tmin=0)
        
            return patterns.plot_topomap(
                times=np.arange(self.components), ch_type='eeg', units='Patterns (arbitrary unit)', 
                          time_unit='s', layout=layout, cmap='RdBu_r', sensors=True,
                          colorbar=True, scalings=None, res=64,
                          size=1.5, cbar_fmt='%3.1f', #name_format='CSP%01d',
                          show=True, show_names=False, title=title, mask=None,
                          mask_params=None, outlines='head', contours=6,
                          image_interp='bilinear', average=None, head_pos=None, time_format='CSP%01d')
    
        else :
            plots = []
            for band in range(self.patterns_.shape[0]) :

                patterns = EvokedArray(self.patterns_[band, :, :].T, info, tmin=0)
                
                thisband = list(self.bands)[band]
    
                plots.append( patterns.plot_topomap(
                    times=np.arange(self.components), ch_type='eeg', units='Patterns (arbitrary unit)', 
                    time_unit='s', layout=layout, cmap='RdBu_r', sensors=True,
                    colorbar=True, scalings=None, res=64,
                    size=1.5, cbar_fmt='%3.1f', #name_format='CSP%01d',
                    show=True, show_names=False, title=(title+' band '+thisband + ' ' + str(self.bands.get(thisband)) + 'Hz'), mask=None,
                    mask_params=None, outlines='head', contours=6,
                    image_interp='bilinear', average=None, head_pos=None, time_format='CSP%01d'))
            return plots
    
    def plot_filters(self, info, layout=None, title=None) :
        from mne import EvokedArray
        import copy as cp
        
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        
        
        if len(self.bands) < 2 :
            patterns = EvokedArray(self.filters_.T, info, tmin=0)
        
            return patterns.plot_topomap(
                times=np.arange(self.components), ch_type='eeg', units='Patterns (arbitrary unit)', 
                          time_unit='s', layout=layout, cmap='RdBu_r', sensors=True,
                          colorbar=True, scalings=None, res=64,
                          size=1.5, cbar_fmt='%3.1f', #name_format='CSP%01d',
                          show=True, show_names=False, title=title, mask=None,
                          mask_params=None, outlines='head', contours=6,
                          image_interp='bilinear', average=None, head_pos=None, time_format='CSP%01d')
    
        else :
            plots = []
            for band in range(self.filters_.shape[0]) :

                patterns = EvokedArray(self.filters_[band, :, :].T, info, tmin=0)
                
                thisband = list(self.bands)[band]
    
                plots.append( patterns.plot_topomap(
                    times=np.arange(self.components), ch_type='eeg', units='Patterns (arbitrary unit)', 
                    time_unit='s', layout=layout, cmap='RdBu_r', sensors=True,
                    colorbar=True, scalings=None, res=64,
                    size=1.5, cbar_fmt='%3.1f', #name_format='CSP%01d',
                    show=True, show_names=False, title=(title+' band '+thisband + ' ' + str(self.bands.get(thisband)) + 'Hz'), mask=None,
                    mask_params=None, outlines='head', contours=6,
                    image_interp='bilinear', average=None, head_pos=None, time_format='CSP%01d'))
            return plots

class Preproc :
    
    def __init__(self, bands=None, classes=2, fs=100, windowlength=3.5, source='raw'):
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
        self.source = source
        
        if bands == None :
            self.bands = {'sub-band' : (7, 30)}
        elif bands == 'all' :
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 20), 'beta-2': (20,30), 'gamma': (30, 45)}
        else:
            self.bands = bands
    
    def center_data(self, raw_data) :
        
        #center data. remove mean from each channel.
        raw_data = np.array([(raw_data[channel] - np.mean(raw_data[channel]))  for channel in range(raw_data.shape[0])])
        print("Centering complete. Centered data shape: " + str(raw_data.shape))
        self.centered_data_=raw_data
        return self.centered_data_
        
    def bpfilter_data(self, raw_data) :
        if len(self.bands.items()) > 1 and self.source!='epoched':
            #band-pass filter raw data
            raw_data_filtered = np.zeros((len(self.bands.items()), raw_data.shape[0], raw_data.shape[1]))
        
            for channel in range(raw_data.shape[0]) :
                i = 0
                for name, band in self.bands.items() :
                    raw_data_filtered[i, channel, :] = filter_data(raw_data[channel], self.fs,  band[0], band[1], copy=True,verbose=0)
                    i += 1
            raw_data = raw_data_filtered
            print("Bandpass filtering complete. Bandpass filtered data shape: " + str(raw_data.shape))
            self.bpfiltered_data_ = raw_data
            
        elif len(self.bands.items()) > 1 and self.source == 'epoched' :
            bpfilter_result = np.zeros((len(self.bands.items()), raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]))
            i = 0
            for key, value in self.bands.items() :
                for epoch in range(raw_data.shape[0]) :
                    for channel in range(raw_data.shape[1]) :
                        bpfilter_result[i,epoch, channel, :] = filter_data(raw_data[epoch, channel, :], sfreq=self.fs,  l_freq=value[0], h_freq=value[1], copy=True, verbose=0, filter_length=(self.fs*self.windowlength), method='iir') 
                i += 1
            self.bpfiltered_data_ = raw_data
        
        else :
            for key, value in self.bands.items() :
                raw_data = np.array([filter_data(raw_data[channel], self.fs,  value[0], value[1], copy=True, verbose=0) for channel in range(raw_data.shape[0])])
            print("Bandpass filtering complete. Bandpass filtered data shape: " + str(raw_data.shape))
            self.raw_data= bpfilter_result
        
        return self.bpfiltered_data_
     
    def epoch(self, raw_data, label_pos) :
        
        #extract epochs         
        samples_per_epoch = int(self.fs*self.windowlength)     #number of samples in a window
        
        if len(self.bands) > 1 :
            
            self.epoched_data_ = np.zeros((len(self.bands.items()), len(label_pos), raw_data.shape[1], samples_per_epoch)) #filter-bank time domain signal with epochs
            
            for epoch in range(len(label_pos)) :
                self.epoched_data_[:, epoch, :, :] = raw_data[:,:,label_pos[epoch]:(label_pos[epoch]+samples_per_epoch)]
            print("Epoching complete. Epoched data shape: " + str(self.epoched_data_.shape))
            
        else :
            self.epoched_data_ = np.zeros((len(label_pos), raw_data.shape[0], samples_per_epoch)) #time domain signal with epochs for a single band.
            
            for epoch in range(len(label_pos)) :
                self.epoched_data_[epoch, :, :] = raw_data[:,label_pos[epoch]:(label_pos[epoch]+samples_per_epoch)]
            print("Epoching complete. Epoched data shape: " + str(self.epoched_data_.shape))
        return self.epoched_data_
                
    def preproc(self, raw_data, label_pos) :
        raw_data = raw_data.astype(float) #filter_data requires float64 unfortunately, therefore int16 needs to be cast to f64
        
        #center data. remove mean from each channel.
        self.center_data(raw_data) # maybe sklearn.preprocessing.scale better?        
        
        #band-pass filter raw data
        self.bpfilter_data(self.centered_data_)
               
        #extract epochs         
        self.epoch(self.bpfiltered_data_, label_pos)
        
        print("Pre-processing complete. Data shape: " + str(self.epoched_data_.shape))
        
        return self.epoched_data_

if __name__ == '__main__':   
    #define parameters
    EEG_DATA_FILEPATH = "..\\..\\..\\eeg-data\\data_set_IVa_aa_mat\\100Hz\\data_set_IVa_aa.mat" #filepath to the data
    #window_length = float(sys.argv[1]) #seconds. How long was the cue displayed? 
    #fs = int(sys.argv[2]) #Hz
    
    window_length = 3.5 #seconds. How long was the cue displayed? 
    fs = 100 #Hz
    
    #read data
    data_set_IVa_aa = loadmat(EEG_DATA_FILEPATH)

    raw_data = data_set_IVa_aa['cnt'].T
    label_pos = data_set_IVa_aa['mrk']['pos'][0][0][0] #epoch position (onset) within time signal
    label_class  = data_set_IVa_aa['mrk']['y'][0][0][0] #epoch class
    
    label_class = label_class[~np.isnan(label_class)] #remove nan's
    label_pos = label_pos[:168] #remove nan's
    
    p = Preproc(bands=None, fs=fs, windowlength=window_length)
    
    #X = p.preproc(raw_data, label_pos)
    X = p.center_data(raw_data)
    X= p.epoch(X, label_pos)
  
    csp = FBCSP(filter_target='raw')
    
    
    #csp = FBCSP(filter_target='raw')
    #print('------------------------------------------------------------')
    #csp.fit(X, label_class)
    #result_1 = csp.transform(raw_data)
    #print('------------------------------------------------------------')
    #csp2 = FBCSP(filter_target='epoched', method='concatenate')
    #result_2 = csp2.fit_transform(X, label_class)
    #print('-------------------------------------------------------------')
    #csp4 = FBCSP(filter_target='epoched', method='avg_amplitude')
    #result_4 = csp4.fit_transform(X, label_class)
    print('-------------------------------------------------------------')
    csp5 = FBCSP(filter_target='epoched', method='avg_power', bands=None, sum_class=True)
    result_5 = csp5.fit_transform(X, label_class)
    
    from mne.decoding import CSP
    csp = CSP(cov_est='epoch')
    csp.fit(X, label_class)
    
    #sf = feature_selector(features=2)
    #sf.fit(result_5, label_class)
    #new_filter = sf.transform(csp5.filters_)

    #from classifier import classifier
    
    #classifier = classifier()
    #classifier.fit(X, label_class)
    
    
    
    """
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import RepeatedKFold, learning_curve #various algorithms to cross validate and evaluate. Maybe RepeatedKFold as well? 
    from sklearn.pipeline import Pipeline
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    from mne.decoding import CSP
    
    fbcsp2 = FBCSP(filter_target='epoched', concatenate=False, avg_band=True, sum_class=False)
    fbcsp = FBCSP(filter_target='epoched', concatenate=False, avg_band=True)
    
    csp = CSP(cov_est='epoch', reg=None, log=False)
    
    myresult_with_sum_class=fbcsp2.fit_transform(X, label_class)
    myresult=fbcsp.fit_transform(X, label_class)    
    
    cspresult=csp.fit_transform(X, label_class)
    print(cspresult.shape)
    
    
    svm = LinearSVC()
    X, y = shuffle(X, y)
    

    fbpipe = Pipeline([('CSP', fbcsp), ('SVM', svm)])
    csppipe = Pipeline([('CSP', csp), ('SVM', svm)])
    
    cv = RepeatedKFold(10, 5) # repeated Kfold. Number_of_folds x Number_of_Repetitions
    
    train_sizesfb, train_scoresfb, test_scoresfb = learning_curve(fbpipe, X, y, cv=cv, verbose=0)
    train_sizescsp, train_scorescsp, test_scorescsp = learning_curve(csppipe, X, y, cv=cv, verbose=0)
    
    #mean and standard deviation for train scores for FBCSP
    train_scores_meanfb = np.mean(train_scoresfb, axis=1)
    train_scores_stdfb = np.std(train_scoresfb, axis=1)
    
    #mean and standard deviation for test scores for MNE CSP
    test_scores_meancsp = np.mean(test_scorescsp, axis=1)
    test_scores_stdcsp = np.std(test_scorescsp, axis=1)
    
    #mean and standard deviation for train scores for MNE CSP
    train_scores_meancsp = np.mean(train_scorescsp, axis=1)
    train_scores_stdcsp = np.std(train_scorescsp, axis=1)
    
    #mean and standard deviation for test scores for FBCSP
    test_scores_meanfb = np.mean(test_scoresfb, axis=1)
    test_scores_stdfb = np.std(test_scoresfb, axis=1)
    
    #plot learning curve

    # Draw lines
    plt.plot(train_sizesfb, train_scores_meanfb, '--', color="#ff0000",  label="Training score with FBCSP")
    plt.plot(train_sizesfb, test_scores_meanfb, color="#0000ff", label="Cross-validation score with FBCSP")
    
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
    """