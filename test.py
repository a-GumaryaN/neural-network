import numpy as np
from new_system import Dense,Neural_network

dense1=Dense(
    layer_number=1,
    prev_layer_length=2,
    layer_length=3,
    activation="sigmoid",

)

dense2=Dense(
    layer_number=2,
    prev_layer_length=3,
    layer_length=1,
    activation="sigmoid",
)

nn=Neural_network(dense1,dense2,epoch=100,learning_rate=1)

test_value=np.array([1,0])

input_values=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

expected_values=np.array([
    [0],
    [1],
    [1],
    [0],
])

nn.learn(input_values,expected_values)

print(nn.forward_propagation(np.array([1,0])))
