import numpy as np

class Adaline(object):
    
    def __init__(self, lr = 0.1, epochs = 50):
        self.epochs=epochs
        self.lr=lr
    
    def fit(self,input,values):
        self.w_ = np.zeros(1 + input.shape[1])
        self.cost_ = []
        
        for i in range(self.epochs):
            output = self.net_input(input)
            errors = (values-output)
            self.w_[1:] += self.lr * input.T.dot(errors)
            self.w_[0] += self.lr * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
            
        return self 
    
    def net_input(self, input):
        
        """Calculating net input"""
        
        return np.dot(input, self.w_[1:]) + self.w_[0]
    
    def activation(self, input):
        
        """ Computer linear activation"""
        
        return self.net_input(input)
    
    def predict(self, input):
        return  np.where(self.activation(input) >= 0.0, 1, -1)