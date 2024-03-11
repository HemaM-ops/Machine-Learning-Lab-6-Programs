from sklearn.neural_network import MLPClassifier
import numpy as np

# AND gate data
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# XOR gate data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Train MLP model for AND gate
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000)
mlp_and.fit(X_and, y_and)

# Train MLP model for XOR gate
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000)
mlp_xor.fit(X_xor, y_xor)

# Test the models
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Predictions for AND gate
predictions_and = mlp_and.predict(test_data)
print("Predictions for AND gate:", predictions_and)

# Predictions for XOR gate
predictions_xor = mlp_xor.predict(test_data)
print("Predictions for XOR gate:", predictions_xor)
