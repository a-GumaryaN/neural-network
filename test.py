import numpy as np
from neural_network  import Dense,Neural_network

dense1=Dense(length=5)
dense2=Dense(length=2)

nn=Neural_network(dense1,dense2,input_num=3)

inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
outputs = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 1]
])


nn.learn(inputs,outputs)

# Test the neural network
for i in range(len(inputs)):
    output = nn.forward_propagation(inputs[i].reshape(1, -1))
    print(f"Input: {inputs[i]}, Predicted: {np.round(output)}, Actual: {outputs[i]}")