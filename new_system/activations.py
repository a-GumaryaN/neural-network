import numpy as np 

def tanh(input):
    return np.tanh(input)

def tanh_derivative(input):
    return 1.-np.tanh(x)**2

def sigmoid(input):
    return 1 / ( 1 + np.exp(input * -1) )

def sigmoid_derivative(input):
    temp=sigmoid(input)
    return temp * ( 1 - temp )
    
def relu(input):
    return 1 / ( 1 + np.exp(input) )

def relu_derivative(input):
    temp=sigmoid(input)
    return temp + ( 1 - temp )

def select_activation (activation_name):
    if activation_name == "sigmoid":
        return sigmoid, sigmoid_derivative
    if activation_name == "tanh":
        return sigmoid, sigmoid_derivative
    elif activation_name == "relu":
        return relu, relu_derivative