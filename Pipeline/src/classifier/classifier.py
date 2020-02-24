from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from feature_selector import feature_selector
import pickle

class classifier :
    
    def __init__(self, components=4, featurebands=4, bands='all', classes=2, estimator='svm', selector='linear'):
        """
        :param components : int. number of CSP components the filter will contain
        :param bands : dict. one or multiple frequency bands
        :param classes : int. number of classes that will be evaluated. Default 2 for binary classification. Multiclass will be evaluated as one-versus-rest.
        :param method : which feature selection algortihm to use. Default: lda, also implemented: svm
        :param prefiltered: has the data been prefiltered or does it need to be filtered?
        
        
        :attribute filters_ : numpy array. CSP filter.  Shape: channels * samples
        """
        self.selector = selector
        self.featurebands = featurebands
        self.estimator = estimator
        self.components  = components
        self.classes = classes
        if bands == None :
            self.bands = {'sub-band' : (7, 30)}
        elif bands == 'all' :
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 20), 'beta-2': (20,30), 'gamma': (30, 45)}
        else:
            self.bands = bands
    
    def load_self(self) :
        self.csp_ = self.load_csp()
        self.filters_ = self.load_filters()
        self.features_ = self.load_features()
        self.classifier_ = self.load_classifier()
    
    def save_filters (self, filters, filename='filters.sav') :
        pickle.dump(filters, open(filename, 'wb'))
        
    def save_csp (self, filters, filename='csp_object.sav') :
        pickle.dump(filters, open(filename, 'wb'))
        
    def save_features (self, features, filename='fbcsp_features.sav') :
        pickle.dump(features, open(filename, 'wb'))
        
    def save_classifier(self, classifier, filename='classifier.sav') :
        pickle.dump(classifier, open(filename, 'wb'))
        
    def load_filters (self, filename='filters.sav') :
        filters = pickle.load(open(filename, 'rb'))
        return filters
    
    def load_csp (self, filename='csp_object.sav') :
        filters = pickle.load(open(filename, 'rb'))
        return filters
    
    def load_features (self, filename='fbcsp_features.sav') :
        features = pickle.load(open(filename, 'rb'))
        return features
        
    def load_classifier (self, filename='classifier.sav') :
        classifier = pickle.load(open(filename, 'rb'))
        return classifier
    
    def fit_raw(self, raw, y, label_pos) :
        preproc = Preproc(fs=161, windowlength=1, bands=self.bands)
        raw_bpfiltered_epoched = preproc.preproc(raw, label_pos)
        preproc = Preproc(fs=161, windowlength=1, bands=None)
        raw_centered = preproc.center_data(raw)
        raw_centered_epoched = preproc.epoch(raw_centered, label_pos)

        self.fit(raw_centered_epoched, raw_bpfiltered_epoched, y)        
        
    
    def fit (self, epoched, X, y) :
        """
        :param X : numpy array of Shape: bands * epochs* channels * samples
        """
        
        fbcsp = FBCSP(filter_target='epoched', method='avg_power', bands=self.bands, components=self.components)
        sf = feature_selector(method=self.selector, features=self.featurebands)
        
        
        #select classifier to use
        if self.estimator == 'lda' :    
            clf = LDA()
        elif self.estimator == 'svm' :
            clf = LinearSVC()
        else :
            print('TODO')
        
    
        
        #fit the fbcsp filters
        fil = fbcsp.fit_transform(X, y)
        
        #run the filters through feature selection and build new filters.
        sf.fit(fil, y)
        new_filt = sf.transform_filter(fbcsp.filters_)
        
        #use the new filter to filter the epoched (same amount of epochs as X) non-bandpass-filtered signal.
        transform_csp = FBCSP(filter_target='epoched', method='avg_power', bands=None)
        transform_csp.filters_ = new_filt
        epoched = transform_csp.transform(epoched)
        
        clf.fit(epoched, y)
        
        self.csp_ = fbcsp
        self.filters_ = sf.transform_filter(fbcsp.filters_)
        self.features_ = sf
        self.classifier_ = clf
        
        self.save_csp(fbcsp)
        self.save_filters(self.filters_)
        self.save_features(sf)
        self.save_classifier(clf)
    
    """
    def transform(self, X) :
        fbcsp = FBCSP(bands=None, filter_target='epoched', method='avg_power')
        fbcsp.filters_ = self.filters_
        return fbcsp.transform(X)
    """
    def score(self, data, y, filterpath='csp_filters.sav', classifierpath='classifier.sav') :
        """
        :param data : numpy array of one window of eeg data. Shape: epochs * channels * samples
        """
        
        #import pdb
        #pdb.set_trace()
        clf = self.classifier_
        sf = self.features_
        fbcsp = self.csp_
        
        fbcsp = FBCSP(bands=None, filter_target='epoched', method='avg_power')
        fbcsp.filters_ = self.filters_
        
        
        filtered = fbcsp.transform(data)
        
            
        score = clf.score(filtered, y)
        return score

    def predict(self, X):
        return self.classifier_.predict(X)