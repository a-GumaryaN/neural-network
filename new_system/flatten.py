import numpy as np
from layer import Layer


class Flatten(Layer):

    input_shape=None
    output_number=None
    delta=None
    layer_type="flatten"

    def __init__(self,input_shape,output_number):
        self.input_shape=input_shape
        self.output_number=output_number

    def calculate(self,input):
        return np.flatten(input)

    def apply_delta(self,delta):
        self.delta = delta

    def calculate_delta(self):
        return self.delta.reshape(self.input_shape)