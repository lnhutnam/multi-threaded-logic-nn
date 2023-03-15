from abc import ABC, abstractmethod

import numpy as np

class ModelExec(ABC):
    def decision_function(self, data):
        return np.dot(data, self.weights[1:3]) - self.weights[0]

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
        X_or_Aug = np.c_[np.ones(X_or.shape[0]), X_or]
        y_or = np.array([0, 1, 1, 1]).reshape(-1, 1)
        self.weights = np.linalg.pinv(X_or_Aug) @ (y_or)
    
    def inference(self, data):
        result = (0, 1)[self.decision_function(data)[0] >= 0.0]
        # print(f'{self.name}({str(data)}) = {result}')
        return result
    
class AND_Perceptron(ModelExec):
    def __init__(self, device='cpu'):
        self.name = 'AND_Perceptron'
        self.device = device
        self.init_weights()

    def init_weights(self, ):
        X_and = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]).T
        X_and_Aug = np.c_[np.ones(X_and.shape[0]), X_and]
        y_and = np.array([0, 0, 0, 1]).reshape(-1, 1)

        self.weights = np.linalg.pinv(X_and_Aug) @ (y_and)
    
    def inference(self, data):
        result = (0, 1)[self.decision_function(data)[0] >= 0.5]
        # print(f'{self.name}({str(data)}) = {result}')
        return result
    
class NOT_Perceptron(ModelExec):
    def __init__(self, device='cpu'):
        self.name = 'NOT_Perceptron'
        self.device = device
        self.init_weights()

    def init_weights(self, ):
        X_not = np.array([0, 1]).reshape(-1, 1)
        self.X = np.c_[np.ones(X_not.shape[0]), X_not]

        y_not = np.array([1, 0]).reshape(-1, 1)
        self.weights = np.linalg.pinv(self.X) @ (y_not)
    
    def inference(self, data):
        result = (0, 1)[self.decision_function(data)[0] >= 0.0]
        # print(f'{self.name}({str(data)}) = {result}')
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