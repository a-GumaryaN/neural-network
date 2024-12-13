import numpy as np

def des(describe,value):
    print(describe)
    print(value)

"""
activations functions with its derivatives
"""

def sigmoid(_,input):
    return 1 / (1 + np.exp(input * -1))

def sigmoid_derivative(_,input):
    return input * ( 1 - input )

def relu(input):
    return np.maximum(input, 0)
    
def relu_derivative(input):
    output=np.full(len(input),0)
    for i in range(len(input)):
        if input[i]>0 :
            output[i]=1
    return output

def MSE(_,actual_value,expected_value):
    error=expected_value-actual_value
    error=error * error
    return error.mean()

def choose_activation(activation_function_name):
    if activation_function_name == "relu":
        return relu,relu_derivative

    if activation_function_name == "sigmoid":
        return relu,relu_derivative

"""
declare dense (layer for fully connected (perceptron) neural network)
"""
class Dense:

    #initial data
    bias=1
    learning_rate=0.1
    weight_matrix=None
    activation=sigmoid
    activation_derivative=sigmoid_derivative
    prev_layer_length=None
    layer_number=None
    layer_length=None
    activation_name="sigmoid"

    #in action data
    layer_output=None
    layer_input=None
    delta=None


    def __init__(self,**keywords):

        self.prev_layer_length=keywords["prev_layer_length"]
        self.layer_number=keywords["layer_number"]
        self.layer_length=keywords["layer_length"]

        if "activation" in keywords:
            self.activation_name=keywords["activation"]
        
        self.activation,self.activation_derivative=choose_activation(self.activation_name)

        if "initial_bias" in keywords:
            self.bias=keywords["initial_bias"]
        
        if "learning_rate" in keywords:
            self.learning_rate

        if "weight" in keywords:
            self.weight_matrix=keywords["weight"]
        else:
            self.weight_matrix=np.random.randn(self.layer_length,self.prev_layer_length)
            #prevent to initialize any weight with zero
            self.weight_matrix = self.weight_matrix + 0.5

    def show(self):
        print("layer ",self.layer_number)
        print("activation :",self.activation_name)
        print("layer input :")
        print(self.layer_input)
        print("weight matrix :")
        print(self.weight_matrix)
        print("bias :",self.bias)
        print("layer delta :")
        print(self.delta)
        print("layer output :")
        print(self.layer_output)

    # this method use for run forward propagation in each layer
    def calculate(self,input):
        self.layer_input=input
        output=self.layer_input.dot(self.weight_matrix.T)
        output= output + self.bias
        output= self.activation(output)
        self.layer_output=output
        return output

    #this method use to get error for each layer and calculate delta for layer
    def calculate_delta(self,error):
        derivative=self.activation_derivative(self.layer_output)
        self.delta=np.multiply(derivative,error)
        return self.delta

    def calculate_delta_matrix(self):
        base_delta_matrix=np.tile(self.delta,(self.prev_layer_length,1))
        delta_matrix=np.multiply(base_delta_matrix,self.weight_matrix.T)
        delta_matrix=np.sum(delta_matrix,axis=1)
        return delta_matrix

    def update_weights(self):
        weight_delta_matrix=np.tile(self.layer_input,(self.layer_length,1))
        delta_matrix=np.tile(self.delta,(self.prev_layer_length,1)).T
        weight_delta_matrix = weight_delta_matrix * self.learning_rate
        weight_delta_matrix = np.multiply(weight_delta_matrix,delta_matrix)
        self.weight_matrix = self.weight_matrix + weight_delta_matrix




class Neural_network:

    #initial data
    error_function=MSE
    topology=None
    layers_number=None
    learning_rate=1
    epoch=1

    #in action data
    network_input=None
    network_output=None
    hidden_layers=[]
    output_layer=None
    input_layer_length=None # length of input layer
    network_error=0


    def __init__(self,*layers,**keywords):

        # first layer length is previous layer length for first hidden layer
        self.input_layer_length=layers[0].prev_layer_length

        # add hidden layers
        for i in range(len(layers)-1):
            self.hidden_layers.append(layers[i])

        # add output layer
        self.output_layer=(layers[-1])

        if "epoch" in keywords:
            self.epoch=keywords["epoch"]
        
    
    def show(self):
        print("---------------------hidden layers---------------------")
        for layer in self.hidden_layers:
            layer.show()
            print("------------------------------------------------------")
        print("---------------------output layer---------------------")
        self.output_layer.show()
        print("------------------------------------------------------")

    def forward_propagation(self,input):
        if len(input) != self.input_layer_length:
            raise Exception("number of input data not equal to number of input layer neurons-->forward propagation process")
        
        temp_input = input
        self.network_input=input

        for layer in self.hidden_layers:
            temp_input=layer.calculate(temp_input)

        self.network_output=self.output_layer.calculate(temp_input)

        return self.network_output


    def learn(self,input_values,expected_values):
        if len(input_values) != len(expected_values):
            raise Exception("number of input values not equal with expected values-->learn function")

        for j in range(self.epoch):
            number_of_data=len(input_values)
            for i in range(number_of_data):
                input_value=input_values[i]
                expected_value=expected_values[i]
                self.backward_propagation(input_value,expected_value)

    def backward_propagation(self,input_value,expected_value):
        #run forward propagation algorithm
        self.forward_propagation(input_value)

        actual_value=self.network_output

        #calculate network error (loss)
        self.network_error=self.error_function(actual_value,expected_value)
        print(self.network_error)

        #calculate output layer error
        error=expected_value - actual_value

        #calculate delta for last layer
        self.output_layer.calculate_delta(error)

        #update all layers weights and biases
        self.output_layer.update_weights()

        #create delta matrix for last layer
        #this matrix is use for calculate delta for previous layer
        error=self.output_layer.calculate_delta_matrix()

        #for each hidden layer calculate delta and delta matrix
        for layer in reversed(self.hidden_layers):
            #calculate delta for each layer
            layer.calculate_delta(error)
            #update weights for each layer
            layer.update_weights()
            #calculate delta matrix for next layer
            error=layer.calculate_delta_matrix()
            