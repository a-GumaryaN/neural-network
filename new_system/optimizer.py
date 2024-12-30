import numpy as np

class Adam:
    t=0
    m=0
    v=0
    beta1=0.9
    beta2=0.999
    epsilon=1e-8
    learning_rate=0.1

    def __init__(self,mv_shape,beta1=0.9,beta2=0.999,epsilon=1e-8,learning_rate=0.1):
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.learning_rate=learning_rate
        self.m=np.zeros_like(mv_shape)
        self.v=np.zeros_like(mv_shape)

    def run(self,param,grad):
        self.t += 1
        self.m = ( self.beta1 * self.m ) + ( (1 - self.beta1) * grad )
        self.v = ( self.beta2 * self.v ) + ( (1 - self.beta2) * (grad ** 2) )
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param + update

class GD:

    learning_rate=0.1

    def __init__(self,learning_rate=0.1):
        self.learning_rate=learning_rate

    def run(self,param,grad):
        return param + ( self.learning_rate * grad )

def select_optimizer(opt_name,mv_shape=None,beta1=0.9,beta2=0.999,epsilon=1e-8,learning_rate=0.1):
    if opt_name=="adam":
        return Adam(mv_shape=None,beta1=0.9,beta2=0.999,epsilon=1e-8,learning_rate=0.1)

    return GD(learning_rate)

        

