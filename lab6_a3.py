import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights
W = np.array([10, 0.2, -0.75])

# Input data for AND gate
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Desired output for AND gate
y = np.array([0, 0, 0, 1])

# Learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Maximum number of epochs
max_epochs = 1000

# Error threshold for convergence
convergence_error = 0.002

# Function to calculate step activation
def step_activation(x):
    return 1 if x >= 0 else 0

# Function to train perceptron
def train_perceptron(X, y, W, alpha, max_epochs, convergence_error):
    iterations_to_converge = []

    for alpha in learning_rates:
        # Copy initial weights for each learning rate
        W_copy = np.copy(W)

        for epoch in range(max_epochs):
            error_sum = 0

            for i in range(len(X)):
                # Calculate the predicted output
                prediction = step_activation(np.dot(X[i], W_copy))

                # Calculate the error
                error = y[i] - prediction

                # Update weights
                W_copy = W_copy + alpha * error * X[i]

                # Accumulate the squared error for this sample
                error_sum += error ** 2

            # Calculate the sum-squared error for all samples in this epoch
            total_error = 0.5 * error_sum

            # Check for convergence
            if total_error <= convergence_error:
                iterations_to_converge.append(epoch + 1)
                break

    return iterations_to_converge

# Train the perceptron and record iterations for each learning rate
iterations_to_converge = train_perceptron(X, y, W, learning_rates, max_epochs, convergence_error)

# Plotting learning rates against iterations to converge
plt.plot(learning_rates, iterations_to_converge, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations to Converge')
plt.title('Iterations to Converge vs. Learning Rate')
plt.grid(True)
plt.show()
