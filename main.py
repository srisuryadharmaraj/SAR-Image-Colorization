#main
from os import listdir
from pix2pix_GAN import define_gan, train, define_generator, define_discriminator
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot
from numpy import asarray
from datetime import datetime
# import torch
from tensorflow.keras.backend import clear_session
clear_session()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


height, width = 256, 256


def load_images(_path):
    src_list, tar_list = list(), list()
    s1 = f's1/'
    for filename in listdir(_path + s1):
        # load and resize the image
        print(filename)
        pixels = load_img(_path + f's1/' + filename, target_size=(height, width))
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into satellite and map
        src_list.append(pixels)
    
    s2 = f's2/'
    for filename in listdir(_path + s2):

        # load and resize the image
        pixels = load_img(_path + f's2/' + filename, target_size=(height, width))
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into satellite and map
        tar_list.append(pixels)

    return [asarray(src_list), asarray(tar_list)]


def preprocess_data(_data):
    # load compressed arrays
    # unpack arrays
    X1, X2 = _data[0], _data[1]
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


path = r'dataset/train_1/'
# load dataset
# src_images = canny_preprocess(path, canny=False)
#
# tar_images = canny_preprocess(path)

[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)

n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()
pyplot.close()

# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]
# define the models

# d_model = define_discriminator(image_shape)
# g_model = define_generator(image_shape)
d_model = load_model('D_model/C3_256d_000042000.keras')
g_model = load_model('G_model/C3_256g_000042000.keras')


# define the composite model

# g_model.summary()
# d_model.summary()

gan_model = define_gan(g_model, d_model, image_shape)

# Define data
# load and prepare training Images
data = [src_images, tar_images]

dataset = preprocess_data(data)

start1 = datetime.now()

train(d_model, g_model, gan_model, dataset, n_epochs=20, n_batch=1)

stop1 = datetime.now()

execution_time = stop1 - start1
print("Execution time is: ", execution_time)
