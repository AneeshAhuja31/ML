import numpy as np

class LRWithMulVarModel:
    def __init__(self,n_iter,alpha):
        self.n_iter = n_iter
        self.alpha = alpha
        self.weights = 0
        self.bias = 0
        self.costlist = []
        self.std = 0
        self.mean = 0
    
    def Predict(self,X):
        return np.dot(X,self.weights) + self.bias
    
    def ComputeCost(self,X,Y):
        m = len(Y)
        Y_bar = self.Predict(X)
        return (1/(2*m))*np.sum((Y_bar-Y)**2)
    
    def Scale(self, X, scale=True):
        if scale == True: #only scale features by mean and std deviation of training feature (dont use mean and std deviation of testing data)
            # Scaling the features by subtracting mean and dividing by standard deviation
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            
            # Prevent division by zero
            self.std[self.std == 0] = 1
        
        X_scaled = (X - self.mean) / self.std
        return X_scaled


    def Fit(self,X,Y):
        m,n = X.shape
        self.weights = np.zeros((n,1))
        X = self.Scale(X)
        
        for i in range(self.n_iter):
            Y_bar = self.Predict(X)
            dw = self.alpha*(1/m)*np.dot(X.T,(Y_bar-Y))
            db = self.alpha*(1/m)*np.sum((Y_bar-Y))
            
            self.weights -= dw
            self.bias -= db

            if i % 10000 == 0:  # Print cost every 100 iterations to track progress
                cost = self.ComputeCost(X, Y)
                self.costlist.append(cost)
                print(f"Iteration {i}, Cost: {cost}")
    
    def MSE(self,Y_bar,Y):
        m = len(Y)
        return (1/m)*np.sum((Y_bar-Y)**2)

