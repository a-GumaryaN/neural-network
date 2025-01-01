import numpy as np
from activations import select_activation
from initializer import select_initializer
from layer import Layer

class Convolution(Layer):

    kernels=None
    bias=None
    """
    shapes of kernels
    """
    kernel_size=None
    """
    depth of layer => number of different model kernels that extract special property
    """
    layer_depth=None
    kernel_optimizer=None
    bias_optimizer=None
    stride=None
    mode="valid"
    layer_type="convolutional"
    input_cols=None
    input_rows=None
    input_depth=None

    def __init__(self,filters,kernel_size=3,stride=1,padding=(0,0),mode="valid",activation="relu",initializer="random"):
        self.layer_depth=filters
        self.activation,self.activation_derivative=select_activation(activation)
        self.initializer=select_initializer(initializer)

        #initiate kernel
        if type(kernel_size) =="tuple":
            self.kernel_size=np.array(kernel_size)
        else:
            self.kernel_size=np.array([kernel_size,kernel_size])

        #initiate padding size
        if mode == "full":
            self.padding = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        elif mode == "valid":
            self.padding = (0, 0)

        #initiate stride
        if type(stride) =="tuple":
            self.stride=np.array(stride)
        else:
            self.stride=np.array([stride,stride])
        
    def init_params(self,layer_number,prev_layer_shape):
        self.layer_number=layer_number
        self.input_rows=prev_layer_shape[0]
        self.input_cols=prev_layer_shape[1]
        self.input_depth=prev_layer_shape[2]

        #calculate shape of data that layer will output
        output_rows = (self.input_rows + ( 2 * self.padding[0] ) - self.kernel_size[0]) // self.stride[0] + 1
        output_cols = (self.input_cols + ( 2 * self.padding[1] ) - self.kernel_size[1]) // self.stride[1] + 1
        self.layer_shape=(output_rows,output_cols,self.layer_depth)

        #initiate bias depend on layer shape
        self.bias = np.random.randn(*self.layer_shape)

        #initiate kernels depend on layer depth, input depth and kernel size
        kernel_shape=(self.input_depth,self.layer_depth,self.kernel_size[0],self.kernel_size[1])
        self.kernels=np.random.randn(*kernel_shape)#=>this should change to initializer function in next commit

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

    def add_padding(self,input):
        pad_row, pad_col = self.padding
        return np.pad(
            input,
            (pad_row, pad_col),
            mode="constant",
            constant_values=0
        )

    def convolution2d(self,input_matrix,kernel):
        output_rows=self.layer_shape[0]
        output_cols=self.layer_shape[1]
        output = np.zeros((output_rows, output_cols))

        input_rows, input_cols = input_matrix.shape
        kernel_rows, kernel_cols = kernel.shape
        stride_row, stride_col = self.stride
        for i in range(0, output_rows):
            for j in range(0, output_cols):
                row_start = i * stride_row
                col_start = j * stride_col
                row_end = row_start + kernel_rows
                col_end = col_start + kernel_cols
                output[i, j] = np.sum(
                    input_matrix[row_start:row_end, col_start:col_end] * kernel
                )
        
        return output

    def calculate(self,input):
        self.layer_input=self.add_padding(np.array(input))
        self.layer_output = np.copy(self.bias)

        for i in range(self.input_depth):
            for j in range(self.layer_depth):
                self.layer_output[i,j] += self.convolution2d(input[i],self.kernels[j])
        
        return self.layer_output


    def apply_delta(self,delta):
        self.output_gradient=delta

    def calculate_delta(self):
        for i in range(self.depth):
            for j in range(self.prev_layer_shape[2]):
                self.kernels_gradient[i, j] = signal.correlate2d(self.input[j], self.output_gradient[i], "valid")
                self.input_gradient[j] += signal.convolve2d(self.output_gradient[i], self.kernels[i, j], "full")
        
    def update_param(self,input):
        self.kernels = self.kernel_optimizer.run(self.kernels,self.kernels_gradient)
        self.bias = self.bias_optimizer.run(self.bias,self.output_gradient)
