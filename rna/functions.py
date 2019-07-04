import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_der(x):
    return  x * (1 - x)

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_der(x):
    return (1 - x^2)