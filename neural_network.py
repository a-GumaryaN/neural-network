import numpy as np
from loss_functions import select_loss_function

class Neural_network:
    network_input=None
    network_output=None
    layers=[]
    input_shape=None
    loss_function=None


    def __init__(self,*layers,input_shape):

        self.input_shape=input_shape
        prev_layer_shape=self.input_shape

        for i in range(len(layers)):
            layers[i].init_params(i+1,prev_layer_shape)
            self.layers.append(layers[i])
            prev_layer_shape=layers[i].layer_shape

    def forward_propagation(self,input):
        self.network_input=input
        temp_input=input

        for layer in self.layers:
            temp_input=layer.calculate(temp_input)
        self.network_output=temp_input

        return temp_input

    def backward_propagation(self,expected_value):
        # error = expected_value - self.network_output
        error = self.loss_function.loss_derivative(self.network_output , expected_value)
        
        for layer in reversed(self.layers):
            layer.apply_delta(error)
            error=layer.calculate_delta()

        for layer in reversed(self.layers):
            layer.update_param()

    def learn(
        self,
        inputs,
        expected,
        epochs=1000,
        optimizer="gd",
        loss_function="mse",
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
        ):
        if len(inputs) != len(expected):
            raise Exception("input values has not equal length with expected values-->learn function")

        #initialize network loss functions
        self.loss_function=select_loss_function(loss_function)

        #initialize layers optimizers
        for layer in self.layers:
            layer.set_optimizer(
                optimizer,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            )

        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_value=inputs[i].reshape(1, -1)
                expected_value=expected[i].reshape(1, -1)

                self.forward_propagation(input_value)

                self.loss_function.loss(self.network_output,expected_value)
                
                self.backward_propagation(expected_value)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss_function.real_error}")
                
        # for epoch in range(epochs):
        #     actual_values=[]
        #     for i in range(len(inputs)):
        #         input_value=inputs[i].reshape(1, -1)
        #         expected_value=expected[i].reshape(1, -1)
        #         self.forward_propagation(input_value)
        #         actual_values.append(self.network_output)
        #         self.backward_propagation(expected_value)

        #     if epoch % 1000 == 0:
        #         total_loss = np.mean(
        #             np.square(
        #                 expected - np.array(actual_values).squeeze()))
        #         print(f"Epoch {epoch}, Loss: {total_loss}")