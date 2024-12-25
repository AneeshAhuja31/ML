from LinearRegressionModel import LRModel_For_1_variable 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:,np.newaxis,2]
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y = diabetes.target
diabetes_Y_train = diabetes_Y[:-30]
diabetes_Y_test = diabetes_Y[-30:]

model = LRModel_For_1_variable(0.000001,1000)
model.Fit(diabetes_X_train,diabetes_Y_train)

diabetes_Y_predict = model.Predict(diabetes_X_test)
mse = model.Mean_Square_Error(diabetes_Y_predict,diabetes_Y_test)

print("Mean Square Error is: ",mse)
print("Weights: ",model.weight)
print("Intercepts",model.bias)

# parameters are x and y axis for scatterplot and plot
plt.scatter(diabetes_X_test,diabetes_Y_test) # first make scatterplot for testing values (y)
plt.plot(diabetes_X_test,diabetes_Y_predict) # then make plot for predicted values
plt.show()