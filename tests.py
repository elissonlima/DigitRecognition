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
    tst = list(map(lambda x: np.append([1], x), net.layers_outputs[:-1]))
    tst.append(net.layers_outputs[-1])
    print(tst)
    #print(net.process_input(np.array([0, 1])))
    #print(net.process_input(np.array([1, 0])))
    #print(net.process_input(np.array([1, 1])))

def testar_backprop():

    net = rna.RNA(learning_rate=0.5)
    net.add_layer(2, "SIG")
    net.add_layer(2, "SIG")
    net.add_layer(1, "LIN")

    for i in range(1000):
        entradas = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
        saidas = [0,1,1,0]

        for i in range(len(entradas)):
            net_output = net.process_input(entradas[i])
            #print("Entrada: " + str(entradas[i]) + " - Saída: " + str(net_output))
            print(net_output)
            if net_output[0] != saidas[i]:
                #print("Saída incorreta, treinando rede")
                net.backpropagation(net_output, saidas[i])
            #else:
            #    print("Rede acertou!" + "Entrada: " + str(entradas[i]) + " - Saída: " + str(net_output ))

if __name__ == "__main__":

    #testar_criacao_rna()
    #testar_input()
    testar_backprop()