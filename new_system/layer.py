class Layer:
    layer_input=None
    layer_output=None
    layer_number=None
    layer_type=None
    optimizer=None
    initializer=None

    def calculate(self,input):
        pass

    def apply_delta(self,delta):
        pass

    def calculate_delta(self):
        pass
    
    def update_param(self):
        pass
