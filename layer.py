class Layer:
    layer_input=None
    layer_output=None
    layer_number=None
    """
    shape of previous layer:
    this is how input of layer looks like:
        for convolutional layer is shape of input matrix that can be input of network or output
            of previous convolutional layer

        for dense layer is number of input of perceptron or number of output for
            previous dense of flatten layer
    """
    prev_layer_shape=None
    """
    shape of output data of current layer
        for convolutional layer is a matrix
        for dense layer is a array
        for flatten layer is a array
    """
    layer_shape=None
    """
    type of layer that can be:
        convolutional
        dense
        flatten
    """
    layer_type=None
    optimizer=None
    initializer=None

    """
    activation function and activation derivative of layer
    """
    activation=None
    activation_derivative=None

    def calculate(self,input):
        pass

    def apply_delta(self,delta):
        pass

    def calculate_delta(self):
        pass
    
    def update_param(self):
        pass
