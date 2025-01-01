import numpy as np

class Error:
    error=0
    number_of_data=0
    real_error=0

class MSE(Error):

    def loss(self,actual_value,expected_value):
        new_error = expected_value - actual_value
        new_error = new_error * new_error
        self.error += new_error
        self.number_of_data+=1
        self.real_error = self.error / self.number_of_data
        return self.real_error
        
    def loss_derivative(self,actual_value,expected_value):
        return ( 2 / self.number_of_data ) * ( expected_value - actual_value )
        
class MAE(Error):

    def loss(self,actual_value,expected_value):
        self.error += np.absolute( expected_value - actual_value )
        self.number_of_data+=1
        self.real_error = self.error / self.number_of_data
        return self.real_error
        
    def loss_derivative(self,actual_value,expected_value):
        self.number_of_data+=1
        if actual_value > expected_value :
            return self.number_of_data
        else :
            return -1 * self.number_of_data
        
class Huber:

    def loss(self,actual_value,expected_value):
        pass
        
    def loss_derivative(self,actual_value,expected_value):
        pass
        
class Cross_entropy:

    def loss(self,actual_value,expected_value):
        self.error += ( expected_value * numpy.log(actual_value) ) + ( ( 1 - expected_value ) * numpy.log( 1 - actual_value ) )
        self.number_of_data-=1 #==> - ( 1 / n )
        self.real_error = self.error / self.number_of_data
        return self.real_error

    def loss_derivative(self,actual_value,expected_value):
        return ( actual_value - expected_value ) / ( actual_value * ( 1 - actual_value ) )
        
class Categorical_cross_entropy:

    def loss(self,actual_value,expected_value):
        pass
        
    def loss_derivative(self,actual_value,expected_value):
        return -1 * ( expected_value / actual_value )
        
class Kullback_leibler_divergence:

    def loss(self,actual_value,expected_value):
        self.error += expected_value * np.log( expected_value / actual_value )
        self.real_error = self.error
        return self.real_error
        
    def loss_derivative(self,actual_value,expected_value):
        return -1 * ( expected_value / actual_value )

class Hinge:

    def loss(self,actual_value,expected_value):
        self.error += np.maximum( 0 , 1 - ( expected_value * actual_value ) )
        self.number_of_data+=1
        self.real_error = self.error / self.number_of_data
        return self.real_error
        
    def loss_derivative(self,actual_value,expected_value):
        pass
        
class Log_cosh:

    def loss(self,actual_value,expected_value):
        self.error += np.log( np.cosh( actual_value - expected_value ) )
        self.real_error = self.error
        return real_error
        
    def loss_derivative(self,actual_value,expected_value):
        return expected_value - np.tanh( actual_value )
        

class Poisson:

    def loss(self,actual_value,expected_value):
        self.error += actual_value - ( expected_value * np.log( actual_value ) )
        self.real_error = self.error
        return self.real_error
        
    def loss_derivative(self,actual_value,expected_value):
        return 1 - ( expected_value / actual_value )



def select_loss_function(loss_function_name="mse"):

    if loss_function_name == "mae":
        return MAE()
        
    elif loss_function_name == "huber":
        return Huber()

    elif loss_function_name == "cross_entropy":
        return Cross_entropy()
        
    elif loss_function_name == "categorical_cross_entropy":
        return Categorical_cross_entropy()
        
    elif loss_function_name == "kullback_leibler_divergence":
        return Kullback_leibler_divergence()
        
    elif loss_function_name == "hinge":
        return Hinge()
        
    elif loss_function_name == "log_cosh":
        return Log_cosh()
        
    elif loss_function_name == "poisson":
        return Poisson()

    else:
        return MSE()