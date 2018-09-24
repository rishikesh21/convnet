import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten
import os
import struct
import numpy as np
#import CS5242.example_network
from CS5242.example_network import example_network
from CS5242.assign2_utils_p2 import mnist_reader
from keras import optimizers, losses
from CS5242.cifar10 import load_training_data, load_test_data, load_class_names


def mnist_improved_network(input_shape=(28, 28, 1), class_num=10):
    """Example CNN

    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST and (32,32,3) for CIFAR (default: {(28,28,1)})
        class_num {int} -- number of classes. Shoule be 10 for both MNIST and CIFAR10 (default: {10})

    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input(shape=input_shape)

    t = Conv2D(64, (3, 3))(im_input)
    t = Activation('relu')(t)
    t = Conv2D(64, (3, 3))(t)
    t = Activation('relu')(t)

    t = MaxPool2D(pool_size=(2, 2))(t)
    t = Dropout(0.25)(t)

    t = Flatten()(t)

    t = Dense(128)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(class_num)(t)

    output = Activation('softmax')(t)

    model = Model(input=im_input, output=output)

    return model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, class_name = mnist_reader()
    model = mnist_improved_network(input_shape=(28, 28, 1))
    # categorical_crossentropy loss, adam optimizer, classifier accuracy
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x=train_x, y=train_y, batch_size=128,epochs=1, verbose=1)
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print('Test accuracy is {:.4f}'.format(acc))