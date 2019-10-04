import numpy as np

class LogisticRegression:
    def __init__(self):
        self.coef= None #维度
        self.intercept_= None #截距
        self._theta=None #
    
    def _sigmoid(self,x):
        y=1.0/(1.0+np.exp(-x))
        return y
    
    def fit(self,X_train,Y_train,eta=0.01,n_iters=1e4):
        assert X_train.shape[0] == Y_train.shape[0]

        #损失函数
        def J(_theta,X_b,y):
            p_predict=self._sigmoid(X_b.dot(_theta))
            try:
                return -np.sum(y*np.log(p_predict)+(1-y)*np.log(1-p_predict))/len(y)
            except:
                return float('inf')

        #sigmoid梯度导数
        def dJ(theta,X_b,y):
            X = self._sigmoid(X_b.dot(theta))
            return X_b.T.dot(X-y)/len(X_b)

        #模拟梯度下降
        def gradient_descent(X_b,y,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
            theta=initial_theta
            i_iter=0
            while i_iter<n_iters:
                gradient=dJ(theta,X_b,y)
                last_theta=theta
                theta=theta-theta*gradient
                i_iter+=1
                if (abs(J(theta,X_b,y)-J(last_theta,X_b,y))<epsilon):
                    break
                return theta
            X_b = np.hstack([np.ones((len(X_train),1)),X_train])
            initial_theta=np.zeros(X_b.shape[1])
            self._theta=gradient_descent(X_b,Y_train,initial_theta,eta,n_iters)
            self.intercept_=self._theta[1:]
            return self

        def predict_proba(self,X_predict):
            X_b=np.hstack([np.ones((len(X_predict),1)),X_predict])
            return self._sigmoid(X_b.dot(self._theta))

        def predict(self,X_preidt):
            proba=self.predict_proba(X_preidt)
            return np.array(proba>0.5, dtype='int')