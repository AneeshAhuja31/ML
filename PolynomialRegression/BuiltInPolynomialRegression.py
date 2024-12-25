import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample Data
x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1.5, 3.5, 7.0, 12.5, 21.5, 34.0])

# Transform to Polynomial Features
degree = 2
poly = PolynomialFeatures(degree)
x_poly = poly.fit_transform(x)

# Fit the Model
model = LinearRegression()
model.fit(x_poly, y)

# Predictions
y_pred = model.predict(x_poly)

print(poly)
# Evaluation
mse = mean_squared_error(y, y_pred)  # Calculate Mean Squared Error
r2 = model.score(x_poly, y)  # Calculate R-squared
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Visualization
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label=f'Polynomial (degree={degree})')
plt.legend()
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


