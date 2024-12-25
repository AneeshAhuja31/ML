import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MultipleLinearRegressionModel import LRWithMulVarModel

train = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Python\SuperVised_Learning\MultipleLinearRegression\train.csv")
test = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Python\SuperVised_Learning\MultipleLinearRegression\test.csv")

train = train.drop(["Unnamed: 0","Id"],axis=1) #The drop function in pandas is used to remove rows or columns from a DataFrame
test = test.drop(["Unnamed: 0","Id"],axis=1)

train_data = train.values
Y_train = train_data[:,-1].reshape(train_data.shape[0],1) #reshape 2D database to 1D database with n rows and 1 coloumn
X_train = train_data[:,:-1]

test_data = test.values
Y_test = test_data[:,-1].reshape(test_data.shape[0],1)
X_test = test_data[:,:-1]

model = LRWithMulVarModel(100000,0.001)
model.Fit(X_train,Y_train)

X_test = model.Scale(X_test)
Y_predict = model.Predict(X_test)
mse = model.MSE(Y_predict,Y_test)

print("MSE is: ",mse)
print("Accuracy is: ",(1-mse)*100)
print("Weights: ",model.weights)
print("Intercepts",model.bias)

rng = np.arange(0,model.n_iter/10000)
plt.plot(rng,model.costlist) 
plt.show()


