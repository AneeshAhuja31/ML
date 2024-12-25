import mynumpy as np
class LRModel_For_1_variable:
    def __init__(self,alpha,n_iter):
        self.alpha = alpha
        self.n_iter = n_iter
        self.weight = 0
        self.bias = 0
    
    def Predict(self,X):
        return self.weight*X + self.bias #return an array of predicted values Y_bar

    def Cost(self,X,Y):
        m = len(Y)
        Y_bar = self.Predict(X)
        return (1/2*m)*np.sum((Y_bar-Y)**2)  #give the cost J

    def Fit(self,X,Y): #function used to train model
        m = len(Y)
        Y_bar = self.Predict(X)

        for i in range(self.n_iter):
            dw = self.alpha*(1/m)*np.sum((Y_bar-Y)*X) #calculate derivative of weight
            db = self.alpha*(1/m)*np.sum(Y_bar-Y)  # calculate derivative of bias

            self.weight -= dw
            self.bias -= db

    def Mean_Square_Error(self,Y_bar,Y):
        m = len(Y)
        return np.mean((Y_bar-Y)**2)  #return MSE to measure efficiency of model
