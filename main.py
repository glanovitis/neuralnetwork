import numpy as np

np.random.seed(0)

# Input data for the neural network
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


# Create hidden layer (hidden only because we don't define how to tweak, the AI will)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        # Create the weights in the size of the inputs, neurons is the amount of output we want to generate
        # Chose to multiply by 0.10 to keep the values small
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # Create biases as an array of zeroes (this is default behaviour but has to be watched in case of a dead array)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Output the dot product of the inputs with the outputs and the added biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create two layers with random sizes and pass it our sample data X
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
