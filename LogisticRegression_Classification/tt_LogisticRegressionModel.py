from LogisticRegressionModel import LogRegModel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Breast Cancer dataset
data = datasets.load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
degree = 2
alpha = 0.001
n_iter = 10000

model = LogRegModel(alpha,n_iter,degree)
model.Fit(X_train, y_train)

# Evaluate the model
y_pred = model.Predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

plt.plot(range(0,n_iter,100),model.costlist)
plt.title("Cost Function over Iterations: ")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()