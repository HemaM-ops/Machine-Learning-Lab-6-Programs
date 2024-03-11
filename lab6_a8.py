import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    """
    Derivative of the sigmoid activation function
    """
    return sigmoid(x) * (1 - sigmoid(x))

def forward_propagation(x1, x2, weights, bias):
    """
    Forward propagation step
    """
    z = np.dot(weights, [x1, x2]) + bias
    return sigmoid(z)

def backward_propagation(x1, x2, target, weights, bias, learning_rate):
    """
    Backward propagation step
    """
    # Forward propagation
    h = forward_propagation(x1, x2, weights, bias)
    
    # Error calculation
    error = h - target
    
    # Delta calculation
    delta = error * derivative_sigmoid(h)
    
    # Weight delta calculation
    weight_delta = delta * np.array([x1, x2])
    
    # Update weights and bias
    weights -= learning_rate * weight_delta
    bias -= learning_rate * delta
    
    return weights, bias

def XOR_gate(x1, x2, target, weights, bias, learning_rate):
    """
    XOR gate logic using a neural network
    """
    # Perform backward propagation
    weights, bias = backward_propagation(x1, x2, target, weights, bias, learning_rate)
    
    # Forward propagation to get the output
    output = forward_propagation(x1, x2, weights, bias)
    
    return output, weights, bias

# Set learning rate and epochs
learning_rate = 0.05
epochs = 1000

# Initialize weights and bias
weights = np.array([0.5, 0.5])
bias = -1.5

# Training data for XOR gate
inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
targets = np.array([0, 1, 1, 0])  # XOR gate truth table

# Training loop
for epoch in range(epochs):
    # Forward and backward propagation for each training example
    for i, (x1, x2) in enumerate(inputs):
        target = targets[i]
        output, weights, bias = XOR_gate(x1, x2, target, weights, bias, learning_rate)
    
    # Calculate error for the epoch
    error = np.mean(np.square(targets - [forward_propagation(x1, x2, weights, bias) for x1, x2 in inputs]))
    
    # Print error for each epoch (optional)
    print(f'Epoch: {epoch+1}, Error: {error}')
    
    # Break loop if error is below convergence threshold
    if error <= 0.002:
        print('Converged!')
        break

print('Training complete!')
