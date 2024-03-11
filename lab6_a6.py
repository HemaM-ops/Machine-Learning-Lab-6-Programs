import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Data Preparation
data = np.array([
    [20, 6, 2, 1],
    [16, 3, 6, 1],
    [27, 6, 2, 1],
    [19, 1, 2, 0],
    [24, 4, 2, 1],
    [22, 1, 5, 0],
    [15, 4, 2, 1],
    [18, 4, 2, 1],
    [21, 1, 4, 0],
    [16, 2, 4, 0]
])

# Separate features and target
X = data[:, :-1]
y = data[:, -1]

# Add bias term to feature matrix
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

# Step 2: Perceptron Learning
# Initialize weights randomly
np.random.seed(42)
weights_perceptron = np.random.rand(X_bias.shape[1])

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the Perceptron
for epoch in range(1000):
    for i in range(X_bias.shape[0]):
        # Forward pass
        z = np.dot(X_bias[i], weights_perceptron)
        prediction = sigmoid(z)
        
        # Calculate the error
        error = y[i] - prediction
        
        # Backpropagation
        weights_perceptron += 0.01 * error * sigmoid_derivative(prediction) * X_bias[i]

# Step 3: Matrix Pseudo-Inverse
# Compute the pseudo-inverse of the feature matrix
pseudo_inverse = np.linalg.pinv(X_bias)

# Calculate weights using pseudo-inverse
weights_pseudo_inverse = np.dot(pseudo_inverse, y)

# Step 4: Evaluation
# Make predictions using perceptron model
predictions_perceptron = sigmoid(np.dot(X_bias, weights_perceptron)) > 0.5

# Make predictions using pseudo-inverse method
predictions_pseudo_inverse = sigmoid(np.dot(X_bias, weights_pseudo_inverse)) > 0.5

# Compare results
accuracy_perceptron = accuracy_score(y, predictions_perceptron)
accuracy_pseudo_inverse = accuracy_score(y, predictions_pseudo_inverse)

precision_perceptron = precision_score(y, predictions_perceptron)
precision_pseudo_inverse = precision_score(y, predictions_pseudo_inverse)

recall_perceptron = recall_score(y, predictions_perceptron)
recall_pseudo_inverse = recall_score(y, predictions_pseudo_inverse)

f1_perceptron = f1_score(y, predictions_perceptron)
f1_pseudo_inverse = f1_score(y, predictions_pseudo_inverse)

print("Perceptron Learning:")
print("Accuracy:", accuracy_perceptron)
print("Precision:", precision_perceptron)
print("Recall:", recall_perceptron)
print("F1-score:", f1_perceptron)

print("\nMatrix Pseudo-Inverse:")
print("Accuracy:", accuracy_pseudo_inverse)
print("Precision:", precision_pseudo_inverse)
print("Recall:", recall_pseudo_inverse)
print("F1-score:", f1_pseudo_inverse)
