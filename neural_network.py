import numpy as np

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

def choose_activation(activation_name):
    if activation_name == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif activation_name == "relu":
        return relu, relu_derivative

class Dense:
    layer_length=None
    prev_layer_length=None
    layer_number=None
    activation=None
    activation_derivative=None
    weight_matrix=None
    bias=None

    layer_input=None
    layer_activation=None
    layer_output=None

    def __init__(self,**keywords):
        self.layer_length=keywords["length"]
        if "activation" in keywords:
            self.activation,self.activation_derivative = choose_activation(keywords["activation"])
        else :
            self.activation,self.activation_derivative = choose_activation("sigmoid")

    def init_params(self,layer_number,prev_layer_length):
        self.layer_number=layer_number
        self.prev_layer_length=prev_layer_length
        self.weight_matrix=np.random.rand(self.prev_layer_length, self.layer_length)
        self.bias=np.random.rand(1, self.layer_length)
    
    def calculate(self,input):
        self.layer_input = input
        self.layer_activation = np.dot(self.layer_input, self.weight_matrix) + self.bias
        self.layer_output = self.activation( self.layer_activation )
        return self.layer_output

    def apply_delta(self,error):
        self.delta = error * self.activation_derivative(self.layer_activation)

    def calculate_delta(self):
        return np.dot(self.delta, self.weight_matrix.T)

    def update_param(self,learning_rate):
        self.weight_matrix += np.dot(self.layer_input.T, self.delta) * learning_rate
        self.bias += np.sum(self.delta, axis=0, keepdims=True) * learning_rate


class Neural_network:
    network_input=None
    network_output=None
    layers=[]
    input_layer_length=None
    def __init__(self,*layers,**keywords):

        self.input_layer_length=keywords["input_num"]
        prev_layer_length=self.input_layer_length

        for i in range(len(layers)):
            layers[i].init_params(i+1,prev_layer_length)
            self.layers.append(layers[i])
            prev_layer_length=layers[i].layer_length

    def forward_propagation(self,input):
        self.network_input=input
        temp_input=input
        for layer in self.layers:
            temp_input=layer.calculate(temp_input)
        self.network_output=temp_input
        return temp_input

    def learn(self,inputs,expected,epochs=10000,learning_rate=0.1):
        if len(inputs) != len(expected):
            raise Exception("input values has not equal length with expected values-->learn function")
        
        for epoch in range(epochs):
            actual_values=[]
            for i in range(len(inputs)):
                input_value=inputs[i].reshape(1, -1)
                expected_value=expected[i].reshape(1, -1)
                self.forward_propagation(input_value)
                actual_values.append(self.network_output)
                self.backward_propagation(expected_value,learning_rate)

            if epoch % 1000 == 0:
                total_loss = np.mean(
                    np.square(
                        expected - np.array(actual_values).squeeze()))
                print(f"Epoch {epoch}, Loss: {total_loss}")

    def backward_propagation(self,expected_value,learning_rate):
        error = expected_value - self.network_output
        
        for layer in reversed(self.layers):
            layer.apply_delta(error)
            error=layer.calculate_delta()

        for layer in reversed(self.layers):
            layer.update_param(learning_rate)