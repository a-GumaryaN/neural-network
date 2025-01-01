from layer import Layer
import numpy as np
from activations import select_activation
from initializer import select_initializer
from optimizer import select_optimizer

class Dense(Layer):
    

    weight_matrix=None
    bias=None

    layer_activation=None
    weight_optimizer=None
    bias_optimizer=None
    layer_type="dense"

    def __init__(self,length,init_limit=1,activation="sigmoid",initializer="random"):
        self.layer_shape=length
        self.activation,self.activation_derivative = select_activation(activation)
        self.initializer=select_initializer(initializer)

    def init_params(self,layer_number,prev_layer_shape):
        self.layer_number=layer_number
        self.prev_layer_shape=prev_layer_shape
        self.weight_matrix=self.initializer(self.layer_shape,self.prev_layer_shape)
        self.bias=np.random.rand(1, self.layer_shape)

    def set_optimizer(self,
                optimizer,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8):
        self.weight_optimizer=select_optimizer(
                optimizer,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                mv_shape=self.weight_matrix.shape
            )
        self.bias_optimizer=select_optimizer(
                optimizer,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                mv_shape=self.bias.shape
            )
    
    def calculate(self,input):
        self.layer_input = input
        self.layer_activation = np.dot(self.layer_input, self.weight_matrix) + self.bias
        self.layer_output = self.activation( self.layer_activation )
        return self.layer_output

    def apply_delta(self,error):
        self.delta = error * self.activation_derivative(self.layer_activation)

    def calculate_delta(self):
        return np.dot(self.delta, self.weight_matrix.T)

    def update_param(self):
        weight_grad=np.dot(self.layer_input.T, self.delta)
        bias_grad=np.sum(self.delta, axis=0, keepdims=True)

        self.weight_matrix = self.weight_optimizer.run(self.weight_matrix,weight_grad)
        self.bias = self.bias_optimizer.run(self.bias,bias_grad)
