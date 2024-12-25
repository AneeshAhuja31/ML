from PolynomialRegressionModel import PolyRegModel

import numpy as np
import matplotlib.pyplot as plt

# Sample Data
x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1.5, 3.5, 7.0, 12.5, 21.5, 34])

degree = 2
model = PolyRegModel(degree)

model.Fit(x,y) #get coefficients
y_pred = model.Predict(x) #predict values of x

mse = model.mean_squared_error(y,y_pred)
print("Coefficients: ",model.w)
print("Mean Squared Error: ",mse)

plt.scatter(x,y,color='blue',label='Actual Data')
plt.plot(x,y_pred,color='red',label=f'Polynomial Fit (degree={degree})')
plt.legend()
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()