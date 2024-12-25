import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, degree=1, alpha=0.01, n_iter=1000):
        self.degree = degree
        self.alpha = alpha
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0
        self.costlist = []

    def PolynomialFeatures(self, x):
        """Generate polynomial features for input X."""
        #if x.ndim == 1:  # Ensure X is a 2D array
        x = x.reshape(-1, 1)
        poly_features = [x**i for i in range(1, self.degree + 1)]
        return np.hstack(poly_features)

    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def ComputeCost(self, x, y):
        m = x.shape[0]
        x_poly = self.PolynomialFeatures(x)
        y_bar = self.Sigmoid(np.dot(x_poly, self.weights) + self.bias)
        return (-1/m) * np.sum(y * np.log(y_bar + 1e-10) + (1 - y) * np.log(1 - y_bar + 1e-10))

    def Fit(self, x, y):
        """Train the model using gradient descent."""
        x_poly = self.PolynomialFeatures(x)
        m, n = x_poly.shape
        self.weights = np.zeros(n)  # Initialize weights based on number of features
        self.bias = 0
        
        for i in range(self.n_iter):
            z = np.dot(x_poly, self.weights) + self.bias
            y_bar = self.Sigmoid(z)
            
            dw = (1 / m) * np.dot(x_poly.T, y_bar - y)
            db = (1 / m) * np.sum(y_bar - y)
            
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

            if i % 100 == 0:
                cost = self.ComputeCost(x, y)
                print(f"Iteration {i} Cost {cost}")
                self.costlist.append(cost)

    def Predict(self, x):
        """Predict binary outcomes (0 or 1)."""
        x_poly = self.PolynomialFeatures(x)  # Apply polynomial transformation to x
        z = np.dot(x_poly, self.weights) + self.bias  # Fix this line to match dimensions
        y_bar = self.Sigmoid(z)
        return y_bar

# Create synthetic data
np.random.seed(42)
X = np.random.randn(100, 1) * 4
y = (X**2 + np.random.randn(100, 1) * 2 > 0).astype(int).flatten()

# Train logistic regression model
model = LogisticRegression(degree=2, alpha=0.01, n_iter=1000)
model.Fit(X, y)

# Create grid points (dense grid to show decision boundary)
x_min, x_max = X.min() - 1, X.max() + 1
xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)

# Apply polynomial features transformation to the grid points
xx_poly = model.PolynomialFeatures(xx)

# Predict the probabilities for each point in the grid
probs = model.Predict(xx_poly)

# Plot decision boundary
plt.figure(figsize=(10, 6))

# Plot the decision boundary line (where the probability is 0.5)
plt.plot(xx, probs, label="Decision Boundary", color="red")

# Plot the original data points
plt.scatter(X, y, c=y, edgecolors='k', marker='o', cmap="coolwarm", label="Data Points")

# Labeling
plt.xlabel("Feature 1")
plt.ylabel("Target Variable")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()
