import rna.network as rna
import numpy as np

def testar_criacao_rna():
    net = rna.RNA()
    net.add_layer(2, "SIG")
    net.add_layer(2, "SIG")
    net.add_layer(1, "HEV")
    print(net.layers)
    pass

def testar_input():
    net = rna.RNA()
    net.add_layer(2, "SIG")
    net.add_layer(2, "SIG")
    net.add_layer(1, "HEV")

    print(net.process_input(np.array([0, 0])))
    print(net.process_input(np.array([0, 1])))
    print(net.process_input(np.array([1, 0])))
    print(net.process_input(np.array([1, 1])))


def testar_backprop():
    net = rna.RNA(3, [2 ,2, 1])
    output = net.process_input(np.array([1,1]))
    print(output)
    net.backpropagation(output, 0)

if __name__ == "__main__":

    #testar_criacao_rna()
    testar_input()
    #testar_backprop()