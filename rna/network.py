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
        self.layers_outputs = []

    def add_layer(self, num_neurons, activate_function):
        layer = {"act_fun":RNA.functions[activate_function], "num_neurons":num_neurons,
                 "act_fun_de":RNA.functions[activate_function+"_"]}
        if self.nlayers > 0:
            self.layers[self.nlayers-1]["weights"] = 2 * np.random.random_sample((self.layers[self.nlayers-1]["num_neurons"] + 1, num_neurons)) - 1
        self.layers.append(layer)
        self.nlayers += 1

    def process_layer(self, input, lay_num):
        self.layers_outputs.append(input)

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

        self.layers_outputs = []
        result = self.process_layer(input, 1)
        self.layers_outputs[-1] = result
        return result

    def backward(self, layer, sensibility):
        curr_layer = self.layers[layer - 1]
        aux = (self.lr * sensibility * self.layers_outputs[layer - 1])
        new_weigths = curr_layer["weights"] + aux[:,np.newaxis]

        sensibility_next = curr_layer["act_fun_de"](self.layers_outputs[layer - 1]) \
                        * np.sum(curr_layer["weights"] * sensibility[:,np.newaxis])

        self.layers[layer - 1]["weights"] = new_weigths

        if layer != 1:
            self.backward(layer - 1, sensibility_next)
        else:
            return

    def backpropagation(self, net_output, correct_output):

        error = correct_output - net_output

        ##BIAS
        outs_aux = list(map(lambda x: np.append([1], x), self.layers_outputs[:-1]))
        outs_aux.append(self.layers_outputs[-1])
        self.layers_outputs = outs_aux

        sensibility_output = self.layers[self.nlayers-1]["act_fun_de"](self.layers_outputs[-1]) * error
        self.backward(self.nlayers - 1, sensibility_output)






