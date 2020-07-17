import numpy as np

# N is batch size.
# D_in is input dimension.
# H is hidden dimension.
# D_out is output dimension.

N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data N is like the no of inputs and D_in or D-out is dimensions of the input. For
# example np.random.randn(2, 5) will be [[1,2,3,4,5],[1,2,3,4,5]] bit the value is vectors something like [
# -1.09856567e+00 -3.07663078e+00  8.00301822e-01  5.25503125e-01 -2.34905288e-02 -5.42183613e-01  1.16659228e+00
# -1.19370352e+00 -1.82373907e+00 -7.31351587e-01]

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
# This means create weight for all the input that is connected to all the nodes
# Similarly for output nodes and hidden nodes

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y

    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2