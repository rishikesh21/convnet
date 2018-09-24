import numpy as np

np.random.seed(0)
from tensorflow import set_random_seed
from keras import optimizers, losses
set_random_seed(0)

import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from matplotlib import pyplot as plt
from PIL import Image

import pickle as pk


def load_image(path):
    im = Image.open(path)
    im = im.resize((224, 224), Image.ANTIALIAS)
    im = im.convert('L')
    return np.array(im) / 255.0


def save_image(path, arr):
    plt.imsave(path, arr)


def test_model_with_weight(weights, x, conv_shape=(3, 3)):
    """Test the weights you get using input image

    Arguments:
        weights {np.array} -- filter weights obtained from the conv layer, should have shape (3,3,1,1)
        x {np.array} -- test image in shape (B,H,W,C), in our case B=1, H=W=224, C=1

    Returns:
        y_pred -- filtered image
    """
    a = Input(shape=(224, 224, 1))
    b = Conv2D(1, conv_shape, padding='valid', use_bias=False)(a)
    model = Model(inputs=a, outputs=b)
    model.get_layer(index=1).set_weights([weights])
    y_pred = model.predict(x)
    return y_pred


def validate(answer_file):
    with open('/Users/mac/Downloads/CS5242/Release/part1/val.pkl', 'rb') as f:
        val = pk.load(f)
    with open(answer_file, 'rb') as f:
        ans = pk.load(f)
    x = np.expand_dims(np.expand_dims(val['val_img'], 0), 3)
    weights = ans['answer']['filter']
    val_img_output = []
    for i in range(0, 3):
        val_img_output.append(
            test_model_with_weight(np.expand_dims(np.expand_dims(weights[i], 2), 3), x, conv_shape=weights[i].shape))
    plt.set_cmap('gray')
    plt.subplot(231)
    plt.imshow(val_img_output[0][0, :, :, 0])
    plt.subplot(232)
    plt.imshow(val_img_output[1][0, :, :, 0])
    plt.subplot(233)
    plt.imshow(val_img_output[2][0, :, :, 0])
    plt.subplot(234)
    plt.imshow(val['val_img_f'][0][0, :, :, 0])
    plt.subplot(235)
    plt.imshow(val['val_img_f'][1][0, :, :, 0])
    plt.subplot(236)
    plt.imshow(val['val_img_f'][2][0, :, :, 0])
    plt.show()




if __name__ == '__main__':

    with open('/Users/mac/Downloads/CS5242/Release/part1/problems.pkl', 'rb') as f:
        problems = pk.load(f)
    for i in range(3):
     if (i<=1):
         size=3
     else:
         size=5

     im_input = Input(shape=(224, 224, 1))
     a = Conv2D(1, (size, size))(im_input)
     model = Model(inputs=im_input, outputs=a)
     sgd = optimizers.SGD(lr=0.01, nesterov=True)
     model.compile(loss='mean_squared_error', optimizer=sgd)
     x = np.expand_dims(np.expand_dims(problems["img"][i], 0), 3)
     y = np.expand_dims(np.expand_dims(problems["img_f"][i], 0), 3)
     model.fit(x=x, y=y, epochs=5000, verbose=0)
     conv_weights = model.get_layer(index=1).get_weights()[0]
     learned_filter = conv_weights[:,:,0,0]
     fig, ax = plt.subplots()
     plt.xticks([0,1,2])
     plt.yticks([0,1,2])
     cs = plt.imshow(learned_filter)
     cbar = plt.colorbar(cs)
     plt.show()
     plt.savefig("/Users/mac/Downloads/CS5242/Release/part1/"+str(i)+".png")
     if (i==1):
         first_filter=learned_filter
     elif (i==2):
         second_filter = learned_filter
     else:
         third_filter = learned_filter

answer = {

'Name': 'RAJENDRAM RAJENDRAM RISHIKESHAN',
# your Matriculation Number, starting with letter 'A'
'MatricNum': 'A0191569N',
# do check the size of filters
'answer': {'filter': [ first_filter, second_filter, third_filter ] }
}
import pickle as pk
with open('/Users/mac/Downloads/CS5242/Release/part1/A0191569N.pkl', 'wb') as f:
 pk.dump(answer, f)
res=validate('/Users/mac/Downloads/CS5242/Release/part1/A0191569N.pkl')





