import numpy as np
import rna.functions as func

class RNA:

    def __init__(self, learning_rate=0.1):

        self.lr = learning_rate
        #Weights of connection of input layer with first hidden layer
        self.wi = 2 * np.random.random((784, 50)) - 1
        #Weights of connection of first hidden layer with second hidden layer
        self.wh1 = 2 * np.random.random((50, 15)) - 1
        #Weights of connections of second hidden layer with output layer
        self.wh2 = 2 * np.random.random((15, 10)) - 1

    def predict(self, input):

        #Output of layer1
        layer1 = func.tanh(np.dot(input, self.wi))
        #Output of layer2
        layer2 = func.tanh(np.dot(layer1, self.wh1))
        #Output of network
        out = func.tanh(np.dot(layer2, self.wh2))

        return out

    def backpropagation(self, input, answer):

        # Output of layer1
        layer1 = func.tanh(np.dot(input, self.wi))
        # Output of layer2
        layer2 = func.tanh(np.dot(layer1, self.wh1))
        # Output of network
        out = func.tanh(np.dot(layer2, self.wh2))

        error = answer - out
        delta_3 = np.multiply(error, func.tanh_der(out))

        error_2 = np.dot(delta_3, self.wh2.T)
        delta_2 = np.multiply(error_2, func.tanh_der(layer2))

        error_1 = np.dot(delta_2, self.wh1.T)
        delta_1 = np.multiply(error_1, func.tanh_der(layer1))

        self.wh2 += self.lr * np.dot(np.array([layer2]*delta_3.shape[0]).T, delta_3).reshape(self.wh2.shape[0], 1)
        self.wh1 += self.lr * np.dot(np.array([layer1]*delta_2.shape[0]).T, delta_2).reshape(self.wh1.shape[0], 1)
        self.wi += self.lr * np.dot(np.array([input]*delta_1.shape[0]).T, delta_1).reshape(self.wi.shape[0], 1)

        return error


    def train(self, inputs, answers, epochs):

        for i in range(epochs):

            epoch_error = 0

            print("Epoch " + str(i + 1) + "/" + str(epochs) + ": ", end='')
            indexes = np.arange(len(inputs))
            np.random.shuffle(indexes)

            for j in indexes:

                curr_error = self.backpropagation(inputs[j], answers[j])
                epoch_error += (np.sum(curr_error) ** 2) / len(inputs)

            print("RMSE:" + str(np.sqrt(epoch_error)))
