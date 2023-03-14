from abc import ABC, abstractmethod

import numpy as np

class ModelExec(ABC):
    @abstractmethod
    def inference(self, data):
        pass

class OR_Perceptron(ModelExec):
    def __init__(self, device='cpu'):
        self.name = 'OR_Perceptron'
        self.device = device
        self.init_weights()

    def init_weights(self, ):
        X_or = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]).T
        y_or = np.array([0, 1, 1, 1]).reshape(-1, 1)
        X_or_Aug = np.c_[np.ones(X_or.shape[0]), X_or]
        result = np.linalg.pinv(X_or_Aug) @ (y_or)

        #result = np.linalg.pinv(X_or_Aug) @ y_or
        self.weights = result[1:3]
        self.bias = result[0]

    def threshold_or(self, s):
        return (0, 1)[s >= 0.0]
    
    def inference(self, data):
        result = np.dot(data, self.weights) - self.bias
        result = self.threshold_or((result)[0])
        print(f'{self.name}({str(data)}) = {result}')
        return result
    
class AND_Perceptron(ModelExec):
    def __init__(self, device='cpu'):
        self.name = 'AND_Perceptron'
        self.device = device
        self.init_weights()

    def init_weights(self, ):
        X_and = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]).T
        y_and = np.array([0, 0, 0, 1]).reshape(-1, 1)
        X_and_Aug = np.c_[np.ones(X_and.shape[0]), X_and]
        result = np.linalg.pinv(X_and_Aug) @ (y_and)

        #result = np.linalg.pinv(X_or_Aug) @ y_or
        self.weights = result[1:3]
        self.bias = result[0]

    def threshold_or(self, s):
        return (0, 1)[s >= 0.5]
    
    def inference(self, data):
        result = np.dot(data, self.weights) - self.bias
        result = self.threshold_or((result)[0])
        print(f'{self.name}({str(data)}) = {result}')
        return result
    
class NOT_Perceptron(ModelExec):
    def __init__(self, device='cpu'):
        self.name = 'NOT_Perceptron'
        self.device = device
        self.init_weights()

    def init_weights(self, ):
        X_not = np.array([0, 1]).reshape(-1, 1)
        y_not = np.array([1, 0]).reshape(-1, 1)
        X_not_Aug = np.c_[np.ones(X_not.shape[0]), X_not]
        result = np.linalg.pinv(X_not_Aug) @ (y_not)

        #result = np.linalg.pinv(X_or_Aug) @ y_or
        self.weights = result[1:3]
        self.bias = result[0]

    def threshold_or(self, s):
        return (0, 1)[s >= 0.0]
    
    def inference(self, data):
        result = np.dot(data, self.weights) - self.bias
        result = self.threshold_or((result)[0])
        print(f'{self.name}({str(data)}) = {result}')
        return result
    
class XOR_Perceptron(ModelExec):
    def __init__(self, device='cpu'):
        self.name = 'XOR_Perceptron'
        self.device = device
        self.or_perceptron_layer = OR_Perceptron()
        self.and_perceptron_layer = AND_Perceptron()
        self.not_perceptron_layer = NOT_Perceptron()
    
    def inference(self, data):
        and_x = self.and_perceptron_layer.inference(data)
        not_x = self.not_perceptron_layer.inference([and_x])
        or_x = self.or_perceptron_layer.inference(data)
        result = self.and_perceptron_layer.inference([not_x, or_x])
        print(f'{self.name}({str(data)}) = {result}')
        return result