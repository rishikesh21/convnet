import os
import struct
import numpy as np
#import CS5242.example_network
from CS5242.example_network import example_network
from keras import optimizers, losses
from CS5242.cifar10 import load_training_data, load_test_data, load_class_names


def _read(dataset="training", path="."):
    """
    Source: https://gist.github.com/akesling/5358964
    Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    which is GPL licensed.
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def mnist_reader():
    from keras.utils import to_categorical
    train_data_iter = _read(dataset='training', path='/Users/mac/Downloads/CS5242/Release/part2/mnist')
    test_data_iter = _read(dataset='testing', path='/Users/mac/Downloads/CS5242/Release/part2/mnist')
    train_x = [];
    train_y = []
    test_x = [];
    test_y = []

    for d in train_data_iter:
        train_x.append(d[1])
        train_y.append(d[0])
    for d in test_data_iter:
        test_x.append(d[1])
        test_y.append(d[0])

    class_name = list(range(0, 10))
    return np.expand_dims(np.stack(train_x, 0) / 255.0, 3), to_categorical(np.array(train_y)), np.expand_dims(
        np.stack(test_x) / 255.0, 3), to_categorical(np.array(test_y)), class_name


def cifar10_reader():
    train_x, _, train_y = load_training_data()
    test_x, _, test_y = load_test_data()
    class_name = load_class_names()
    return train_x, train_y, test_x, test_y, class_name

batch_size = 128
kernel_size = 3
filters = 64


def train_mnist():
    train_x, train_y, test_x, test_y, class_name = mnist_reader()
    model = example_network(input_shape=(28, 28, 1))
    sgd = optimizers.SGD(lr=0.1, decay=0.1, momentum=0.9, nesterov=False)

    # categorical_crossentropy loss, sgd optimizer, classifier accuracy
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(x=train_x, y=train_y,batch_size=128, epochs=1, verbose=1)
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print('Test accuracy is {:.4f}'.format(acc))
    json_string = model.to_json()
    import json

    with open("/Users/mac/Downloads/CS5242/Release/part1/mnist_model.json", 'w') as outfile:
        json.dump(json_string, outfile)


def train_cifar10():
    train_x, train_y, test_x, test_y, class_name = cifar10_reader()
    model = example_network(input_shape=(32, 32 ,3))
    sgd = optimizers.SGD(lr=0.1, decay=-0.1, momentum=0.9, nesterov=False)

    # categorical_crossentropy loss, Adam optimizer, classifier accuracy
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(x=train_x, y=train_y,epochs=20, verbose=1)
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print('Test accuracy is {:.4f}'.format(acc))
    json_string = model.to_json()
    import json

    with open("/Users/mac/Downloads/CS5242/Release/part1/cifar_model.json", 'w') as outfile:
        json.dump(json_string, outfile)


if __name__ == '__main__':
    #train_mnist()
    train_cifar10()

