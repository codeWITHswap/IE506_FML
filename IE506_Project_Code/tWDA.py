from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from pyriemann.utils.base import logm,_matrix_operator
from joblib import Parallel, delayed
import numpy as np  


from tWishartEstimation import RCG

    
# Classification through t-Wishart  
class tWDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self,n,df,n_jobs=1):
        self.n = n # Number of time samples 
        self.df = df # Degrees of Freedom for the model
        self.n_jobs = n_jobs
        
    
    def compute_class_center(self,S,df): # S is the ndarray of SPD matrices 
        _,p,_ = S.shape # (n_trials, n_channels, )
        if df==np.inf: # For WDA
            return np.mean(S,axis=0)/self.n
        return RCG(S,self.n,df=df)

    # Estimation of the centroids
    def fit(self, S, y):

        self.classes_ = np.unique(y)
        Nc = len(self.classes_)
        
        y = np.asarray(y) # y is the label corresponding to each trial
        p,_ = S[0].shape
        if self.n_jobs==1:
            self.centers = [self.compute_class_center(S[y==self.classes_[i]],self.df) for i in range(Nc)]
        else:
            self.centers = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_class_center)(S[y==self.classes_[i]],self.df) for i in range(Nc))
        self.pi = np.ones(Nc)
        
        for k in range(Nc):
            self.pi[k]= len(y[y==self.classes_[k]])/len(y)
        
        return self
 
    # Predict the distance
    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.centers)
        K,p,_ =covtest.shape
        dist = np.zeros((K,Nc)) 
        
        for i in range(Nc):
            if (self.df==np.inf):
                log_h = lambda t:-0.5*t
            else:
                log_h = lambda t:-0.5*(self.df+self.n*p)*np.log(1+t/self.df)
 
            center = self.centers[i].copy()
            inv_center = _matrix_operator(center,lambda x : 1/x)
            logdet_center = np.trace(logm(center))
            for j in range(K):
                # Distance between the center of class i and the covariance j
                dist[j,i] = np.log(self.pi[i])-0.5*self.n*logdet_center+log_h(np.matrix.trace(inv_center@covtest[j]))
        return dist

    # Retrieving the predictions
    def predict(self, covtest):
        dist = self._predict_distances(covtest)
        preds = []
        n_trials,n_classes = dist.shape
        for i in range(n_trials):
            preds.append(self.classes_[dist[i,:].argmax()])
        preds = np.asarray(preds)
        return preds

    # Retrive the distance to each centroid
    def transform(self, S):
        return self._predict_distances(S)

    # Fit and predict in one function
    def fit_predict(self, S, y):
        self.fit(S, y)
        return self.predict(S)

    # Predict the probability using softmax
    def predict_proba(self, S):
        return softmax(-self._predict_distances(S)**2)




