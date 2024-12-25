import numpy as np

class PolyRegModel:
    
    def __init__(self,degree):
        self.degree = degree
        self.w = None #list of coefficients
    
    def Polynomial_Features(self,x):
        x = x.reshape(-1,1) #ensure x is a column vector
        return np.hstack([x**i for i in range(self.degree + 1)]) #returns design matrix (which is a matrix of degrees of the function)
    
    def Fit(self,x,y):
        X_poly = self.Polynomial_Features(x)
        self.w = np.linalg.inv(X_poly.T@X_poly)@(X_poly.T@y) #update coefficient list

    def Predict(self,x):
        X_poly = self.Polynomial_Features(x)
        return X_poly @ self.w #return predicted values
    
    def mean_squared_error(self,y,y_pred):
        return np.mean((y_pred-y)**2)
