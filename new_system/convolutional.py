import numpy as np
from activations import select_activation
from initializer import select_initializer

class Convolution:

    kernels=None
    bias=None
    depth=None
    """
    shapes of kernels
    """
    layer_shape=None #number of kernels (depth of layer)
    """
    shape of input for convolutional layer 
    (dimensions and chanels of previous layer)=>( dim1 , dim2 , ... , dimn)
    """
    prev_layer_shape=None
    activation=None
    activation_derivative=None
    kernel_optimizer=None
    bias_optimizer=None
    layer_type="convolutional"

    def __init__(self,kernel_shape,activation="relu",initializer="random"):
        self.layer_shape=shape()
        self.activation,self.activation_derivative=select_activation(activation)
        self.initializer=select_initializer(initializer)

        
    def init_params(self,layer_number,prev_layer_shape):
        self.layer_number=layer_number
        self.prev_layer_shape=prev_layer_shape
        # n dimensional initializer for kernels and bias

    def set_optimizer(self,
                optimizer,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8):
        self.kernel_optimizer=select_optimizer(
                optimizer,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                mv_shape=self.kernels[0].shape #all of kernels have one shape in each convolutional layer
            )
        self.bias_optimizer=select_optimizer(
                optimizer,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                mv_shape=self.bias[0].shape #all biases have one shape in each convolutional layer
            )

    def calculate(self,input):
        pass

    def apply_delta(self,delta):
        self.output_gradient=delta

    def calculate_delta(self):
        for i in range(self.depth):
            for j in range(self.input_chanel):
                self.kernels_gradient[i, j] = signal.correlate2d(self.input[j], self.output_gradient[i], "valid")
                self.input_gradient[j] += signal.convolve2d(self.output_gradient[i], self.kernels[i, j], "full")
        
    def update_param(self,input):
        self.kernels = self.kernel_optimizer.run(self.kernels,self.kernels_gradient)
        self.bias = self.bias_optimizer.run(self.bias,self.output_gradient)
