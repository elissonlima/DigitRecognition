import numpy as np


class RNA:


    #Neurons activate functions
    #SIG -> sigmoid logistic
    #HEV -> heaviside
    #LIN -> linear
    #TAN -> hyperbolic tangent

    functions = {
        "SIG" : np.vectorize(lambda x : (1 / (1 + np.exp(-x)))),
        "HEV" : np.vectorize(lambda x : 1 if x >=0 else 0),
        "LIN" : np.vectorize(lambda x : x),
        "TAN" : np.vectorize(lambda x : (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))),
        ##derivates
        "SIG_" : np.vectorize(lambda x : x * (1 - x)),
        "HEV_" : np.vectorize(lambda x : 1),
        "LIN_" : np.vectorize(lambda x : 1),
        "TAN_" : np.vectorize(lambda x : (1 - x^2))
    }

    #nlayers -> layer number
    #netconfig -> configuration of network, e.g., if nlayers=3
    #netconfig = [3,3,1] means that network will have
    #3 neurons in first layer, 3 in hidden layer and 1 in last
    def __init__(self, learning_rate=0.1):
        self.layers = []
        self.nlayers = 0
        self.lr = learning_rate

    def add_layer(self, num_neurons, activate_function):
        layer = {"act_fun":RNA.functions[activate_function], "num_neurons":num_neurons}
        if self.nlayers > 0:
            self.layers[self.nlayers-1]["weights"] = np.random.rand(self.layers[self.nlayers-1]["num_neurons"] + 1, num_neurons)
        self.layers.append(layer)
        self.nlayers += 1

    def process_layer(self, input, lay_num):
        if lay_num == self.nlayers:
            out = self.layers[lay_num - 1]["act_fun"](input)
            return out
        else:
            input = np.append([1], input) ##BIAS
            out_l = np.sum(input[:, np.newaxis] * self.layers[lay_num - 1]["weights"], axis=0)
            out = self.layers[lay_num - 1]["act_fun"](out_l)
            return self.process_layer(out, lay_num + 1)

    def process_input(self, input):

        if input.shape[0] != self.layers[0]["num_neurons"]:
            raise Exception("Incorrect input size. Must be " + str(self.netconfig[0]))

        return self.process_layer(input, 1)

    def backward(self, layer, output):

        pass

    def backpropagation(self, net_output, correct_output):

        error = correct_output - net_output






