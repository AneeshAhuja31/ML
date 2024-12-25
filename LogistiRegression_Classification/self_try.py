import numpy as np
class LogRegModel:
    def __init__(self,alpha,n_iter,degree):
        self.alpha = alpha
        self.n_iter = n_iter
        self.degree = degree
        self.w = None
        self.b = None
        self.costlist =[]

    def PolynomialFeatures(self,x):
        if x.ndim == 1:  # Ensure X is a 2D array
            x = x.reshape(-1, 1)
        return np.hstack([x**i for i in range(1,self.degree+1)])  #return design matrix
    
    def Sigmoid(self,z):
        return 1/(1+np.exp(-z)) #return sigmoid of linear function
    
    def ComputeCost(self,x,y):
        m = x.shape[0]
        x_poly = self.PolynomialFeatures(x)
        z = np.dot(x_poly,self.w) + self.b
        y_bar = self.Sigmoid(z)
        return (-1/m)*np.sum(y*np.log(y_bar + 1e-10)+(1-y)*np.log(1-y_bar + 1e-10))
    
    def Fit(self,x,y):
        x_poly = self.PolynomialFeatures(x)
        m,n = x_poly.shape
        self.w = np.zeros(n) #initialize weights
        self.b=0  #initialize bias
        for i in range(self.n_iter):
            z = np.dot(x_poly,self.w)+self.b  #where z is the linear function inputed into the isgmoid function
            y_bar = self.Sigmoid(z)
            #graadient descent
            dw = (1/m)*np.dot(x_poly.T,y_bar-y)
            db = (1/m)*np.sum(y_bar-y)

            self.w -= self.alpha*dw
            self.b -= self.alpha*db

            if i%100 == 0: 
                cost = self.ComputeCost(x,y)
                print(f"Iteration {i} Cost {cost}")
                self.costlist.append(cost) #append cost list after every 100 iterations
    
    def Predict(self,x):
        x_poly = self.PolynomialFeatures(x)
        z = np.dot(x_poly,self.w)+self.b
        y_bar = self.Sigmoid(z)
        return [1 if i>=0.5 else 0 for i in y_bar]
    
        