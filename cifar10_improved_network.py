import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten
import os
import struct
import numpy as np
#import CS5242.example_network
from CS5242.example_network import example_network
from CS5242.assign2_utils_p2 import cifar10_reader
from keras import optimizers, losses
from CS5242.cifar10 import load_training_data, load_test_data, load_class_names
from keras.preprocessing.image import ImageDataGenerator



def cifar10_improved_network(input_shape=(28, 28, 1), class_num=10):
    """Example CNN

    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST and (32,32,3) for CIFAR (default: {(28,28,1)})
        class_num {int} -- number of classes. Shoule be 10 for both MNIST and CIFAR10 (default: {10})

    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input(shape=input_shape)

    t = Conv2D(32, (3, 3))(im_input)
    t = Activation('relu')(t)
    t = Conv2D(32, (3, 3))(t)
    t = Activation('relu')(t)

    t = MaxPool2D(pool_size=(2, 2))(t)
    t = Dropout(0.25)(t)
    t = Conv2D(32, (3, 3))(t)
    t = Activation('relu')(t)
    t = MaxPool2D(pool_size=(2, 2))(t)
    t = Dropout(0.25)(t)


    t = Flatten()(t)

    t = Dense(512)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(class_num)(t)

    output = Activation('softmax')(t)

    model = Model(inputs=im_input, outputs=output)

    return model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, class_name = cifar10_reader()
    model = cifar10_improved_network(input_shape=(32, 32, 3))
    opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # categorical_crossentropy loss, adam optimizer, classifier accuracy
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    #
    #model.fit(x=train_x, y=train_y, batch_size=32,epochs=1, verbose=1,shuffle=True)
    #loss, acc = model.evaluate(x=test_x, y=test_y)
    #print('Test accuracy is {:.4f}'.format(acc))

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    model.fit_generator(datagen.flow(train_x, train_y,batch_size=32),
                        epochs=1,
                        validation_data=(test_x, test_y),
                        workers=4)
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print('Test accuracy is {:.4f}'.format(acc))

    model.save("A0191569N_cifar10.h5")