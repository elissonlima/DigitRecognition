import rna.network as rna
import os
import cv2
import numpy as np
from PIL import Image


def translate(x):

    if isinstance(x, np.ndarray):
        a = np.nonzero(x==x.max())[0][0]
        return a
    else:
        a = np.zeros((10,)).astype(np.float64)
        np.put(a, x, 1.)
        return a


def get_training_set():

    path_ = os.path.join(".", "samples", "trainingSample")
    answers = [f for f in os.listdir(path_)]
    training_set_path = []

    for a in answers:
        training_set_path = np.append(training_set_path, [os.path.join(path_,a, f) for f in os.listdir(os.path.join(path_, a))])

    np.random.shuffle(training_set_path)
    answers = [translate(int(os.path.basename(os.path.dirname(f)))) for f in training_set_path]
    ##We divide the images by 255 to avoid overflow warnings on sigmoid functions
    training_set = np.array([cv2.imread(f, 0) / 255 for f in training_set_path])
    training_set = training_set.reshape((training_set.shape[0], training_set.shape[1] * training_set.shape[2]))

    return training_set, answers


def get_test_set():

    path_ = os.path.join(".", "samples", "testSample")
    ##We divide the images by 255 to avoid overflow warnings on sigmoid functions
    testSet = np.array([cv2.imread(os.path.join(path_, f), 0) / 255 for f in os.listdir(path_)])
    testSet = testSet.reshape((testSet.shape[0], testSet.shape[1] * testSet.shape[2]))
    return testSet


if __name__ == "__main__":
    net = rna.RNA()

    print("Loading training set... ", end='')
    training_set, answers = get_training_set()
    print("OK")

    print("Loading test set... ", end='')
    testSet = get_test_set()
    print("OK")

    print("Starting training...")
    net.train(training_set, answers, 1000)
    print("Train finished")

    img = Image.fromarray((testSet[0] * 255).astype(np.uint8).reshape((28, 28)))
    img.show()
    out = net.predict(testSet[0])
    print("Net answer: " + str(out))
    print("Net answer translated: " + str(translate(out)))
