import numpy as np

def xavier(layer_length,prev_layer_length):
    limit = np.sqrt(6 / (layer_length + prev_layer_length))
    return np.random.uniform(-limit, limit, size=[prev_layer_length,layer_length])

def random(layer_length,prev_layer_length):
    return np.random.rand(prev_layer_length,layer_length)

def select_initializer(initializer="random"):
    if initializer == "xavier":
        return xavier
    else:
        return random
