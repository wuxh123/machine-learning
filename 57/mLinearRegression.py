import numpy as np
from numpy import linalg

class MLinearRegression:
    def __init__(self):
        self.coef_=None
        self.interception_=None
        self._theta = None
    
    def fit(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0]
        ones=np.ones((X_train.shape[0],1))
        X_b=np.hstack((ones,X_train))
        self._theta=linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self,X_predict):
        ones=np.ones((X_predict.shape[0],1))
        X_b=np.hstack((ones,X_predict))
        return X_b.dot(self._theta)

    def mean_squared_error(self,y_true,y_predict):
        return np.sum((y_true-y_predict)**2)/len(y_true)

    def score(self,X_test,y_test):
        y_predict=self.predict(X_test)
        return 1-(self.mean_squared_error(y_test,y_predict))/np.var(y_test)

