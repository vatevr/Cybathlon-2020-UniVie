from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
import math

class feature_selector :
    
    def __init__(self, features=6, method='linear', drops=None):
        """
        :param features : int. number of CSP components the filter will contain
        """
        self.drops = drops
        self.features  = features
        self.method = method
        
    def fit(self, X_csp, y):
        """
        :param X : numpy array of shape bands *  epochs * csp_components
        """
        
        if self.method == 'mi' :
            self.scores_ = self.mi_based_selection(X_csp, y)
        elif self.method == 'linear' :
            self.scores_ = self.linear_regression_based_selection(X_csp, y)
        else :
            print('TODO')
        
        band_ind = np.flip(np.argsort(self.scores_))
        band_ind = band_ind[:self.features] #pick the best n features
        self.bands_ = band_ind
        print('feature selection scores: ' + str(self.scores_))
        
    """    
    def transform(self, X_csp) :
        
        filtered = np.zeros(( X_csp.shape[2], X_csp.shape[1]*self.features))
        band_ind = self.bands_
        
        for feature in range(self.features) :
            for component in range(X_csp.shape[1]) :
                picks[(feature*X_csp.shape[1])+component, :] = X_csp[band_ind[feature], component, :]
        return picks
    """
    
    def transform_filter(self, csp_filter) :
        """
        :param csp_filters : numpy array of shape bands *  csp_components * channels
        """
        picks = np.zeros(((csp_filter.shape[1]*self.features)-len(self.drops), csp_filter.shape[2]))
        band_ind = self.bands_
        #import pdb
        #pdb.set_trace()
        i = 0
        for feature in range(self.features) :
            for component in range(csp_filter.shape[1]) :
                should_drop = False
                for drop in self.drops :
                    band = drop//10 # floor division to avoid importing math.floor
                    comp = drop%10
                    if feature == band and component == comp:
                        should_drop = True
                if should_drop == True:
                    continue
                else :
                    picks[i, :] = csp_filter[band_ind[feature], component, :]
                    i += 1
        return picks
    
            
    """
    def fit_transform(self, X_csp, y) :
        self.fit(X_csp, y)
        return self.transform_filter(X_csp)
    """
    def mi_based_selection(self, X_csp, y) :
        scores = np.zeros((X_csp.shape[0]))
        
        for band in range(X_csp.shape[0]) :
            scores[band] = np.mean(mutual_info_regression(X_csp[band], y))
            print('Score for frequency band ' + str(band) + ": " + str(scores[band]))
        
        return scores
        
    def linear_regression_based_selection(self, X_csp, y) :
        clf = SVC(kernel="linear")
        scores = np.zeros((X_csp.shape[0]))
        
        for band in range(X_csp.shape[0]) :
            scores[band] = np.mean(cross_val_score(clf, X_csp[band], y))
            print('Score for frequency band ' + str(band) + ": " + str(scores[band]))
        
        return scores