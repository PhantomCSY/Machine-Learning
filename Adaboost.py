import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR

class BoostTreeClassifier:
    def __init__(self, base_classfier, max_iter):
        self.base_classfier = base_classfier
        self.max_iter = max_iter
        self.fitted = False
        
    def fit(self, X, y):
        X, y, self.mapping = input_check(X, y)
        
        # Initialization
        nsample = len(y)
        weight = np.ones(nsample) / nsample
        error_rate = np.ones(self.max_iter)
        alpha = np.ones(self.max_iter)
        classifiers = [None] * self.max_iter
        
        # Estimation
        i = 0
        while i < self.max_iter:
            self.base_classfier.fit(X, y, weight)
            
            classifiers[i] = deepcopy(self.base_classfier)
            y_estimated = classifiers[i].predict(X)
            error_rate[i] = weight[y != y_estimated].sum()
            alpha[i] = 0.5 * np.log((1-error_rate[i]) / error_rate[i])
                
            # Update
            weight[y==y_estimated] = weight[y==y_estimated] * np.exp(-alpha[i])
            weight[y!=y_estimated] = weight[y!=y_estimated] * np.exp(alpha[i])
            weight /= weight.sum()
            
            # Accuracy checking
            y_estimated = np.zeros(nsample)
            for j in range(i+1):
                y_estimated += alpha[j] * classifiers[j].predict(X)
            y_estimated = np.sign(y_estimated)
            if (y != y_estimated).sum() == 0:
                i += 1
                break
            
            i += 1
            
        # Finalize
        self.classifiers = classifiers[:i]
        self.alphas = alpha[:i]
        self.fitted = True
        
    def predict(self, X):
        if self.fitted == False:
            raise Exception("Need to fit model first")
        
        nsample = X.shape[0]
        nclassifier = len(self.classifiers)
        
        y_estimated = np.zeros(nsample)
        for j in range(nclassifier):
            y_estimated += self.alphas[j] * self.classifiers[j].predict(X)
        y_estimated = y_estimated.reshape(-1, 1)
        
        y_estimated = np.hstack([np.sign(y_estimated), y_estimated])
        y_estimated[:, 0] = [self.mapping[item] for item in y_estimated[:, 0]]
        
        return y_estimated
            
            
class BoostTreeRegressor:
    def __init__(self, base_regressor, max_iter):
        self.base_regressor = base_regressor
        self.max_iter = max_iter
        self.fitted = False
        
    def fit(self, X, y):
#         X, y, self.mapping = input_check(X, y)
        
        # Initialization
        nsample = len(y)
        regressors = [None] * self.max_iter
        
        # Estimation
        i = 0
        residuals = y.copy()
        while i < self.max_iter:
            self.base_regressor.fit(X, residuals)
            
            regressors[i] = deepcopy(self.base_regressor)
            y_estimated = np.zeros(nsample)
            for j in range(i+1):
                y_estimated += regressors[j].predict(X)
                
            # Accuracy checking
            if (y != y_estimated).sum() == 0:
                i += 1
                break
                
            # Update
            residuals = y - y_estimated
            i += 1
            
        # Finalize
        self.regressors = regressors[:i]
        self.fitted = True
        
    def predict(self, X):
        if self.fitted == False:
            raise Exception("Need to fit model first")
        
        nsample = X.shape[0]
        nregressor = len(self.regressors)
        
        y_estimated = np.zeros(nsample)
        for j in range(nregressor):
            y_estimated += self.regressors[j].predict(X)
        y_estimated = y_estimated
        
        return y_estimated
            
            
def input_check(X, y):
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)
        
    if (not isinstance(X, np.ndarray)) and (not isinstance(y, np.ndarray)):
        raise Exception("X and y should be list or numpy.array")
        
    if len(X.shape) != 2:
        raise Exception("X should be a 2d-array")
    if len(y.shape) != 1:
        raise Exception("y should be an 1d-array")
    if X.shape[0] != len(y):
        raise Exception("The length of X and y are mismatched, with rows of " + str(X.shape[0]) + " and " + str(len(y)))
    if len(np.unique(y)) > 2:
        raise Exception("Only support 2-class classification")
        
    mapping = {}
    classes = np.unique(y)
    y_ = y.copy()
    map_class = [-1, 1]
    for i in range(2):
        mapping[map_class[i]] = classes[i]
        y_[y==classes[i]] = map_class[i]
        
    return (X, y_, mapping)
