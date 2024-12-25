#MatPlotLib library for creating visualizations, such as plots, histograms, and bar charts
import matplotlib.pyplot as plt  #PyPlot is a module of MatPlotLib that makes visualizations
import mynumpy as np   #NumPy provides support for a large, multidimensional arrays and matrices
from sklearn import datasets, linear_model   # Scikit-Learn is an open source library that simplifies the process of implementing ML and statistical models. 

diabetes = datasets.load_diabetes() #in sklearn, dataset modules provide functions to load/gerate several standard datasets which help in training and testing ML models

#diabetes dataset is a dictionary-like object that contains the data (diabetes.data) and target values (diabetes.target)
np.shape(diabetes.data)  #OUTPUT: (442,10) 442 samples and 10 features in this dataset
diabetes_X = diabetes.data[:, np.newaxis,2] # :(Row Slicing) selects all rows ,2 selects the 3rd feature(indexed by 2) from all rows (which is BMI)

diabetes_X_train = diabetes_X[:-30] # this selects all elements of datasets except the last 30 for training
diabetes_X_test = diabetes_X[-30:]  # this selects the last 30 elements of the dataset for testing

# following will split the target variable(Y-axis) into training and testing variable
diabetes_Y = diabetes.target

diabetes_Y_train = diabetes_Y[:-30] #this selects all but the last 30 samples for training
diabetes_Y_test = diabetes_Y[-30:]  #this selects the last 30 elements of the dataset for testing (diabetes_Y_test is denoted as y mathematically)

model = linear_model.LinearRegression() #creates a new linear regression model form skikit-learn to learn from data

model.fit(diabetes_X_train,diabetes_Y_train) #.fit train the model using the training data

diabetes_Y_predicted = model.predict(diabetes_X_test) #uses the linear regression model to predict the y_target values (y bar)

from sklearn.metrics import mean_squared_error
print("Mean Square Error is: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted)) #we get this value after applying cost function on diabetes_Y_predicted which is can also be called y bar
print("Weights: ",model.coef_)
print("Intercepts",model.intercept_)

# parameters are x and y axis for scatterplot and plot
plt.scatter(diabetes_X_test,diabetes_Y_test) # first make scatterplot for testing values (y)
plt.plot(diabetes_X_test,diabetes_Y_predicted) # then make plot for predicted values
plt.show()