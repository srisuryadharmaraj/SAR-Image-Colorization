import numpy as np
# import torch
from keras.models import load_model
from matplotlib import pyplot
from numpy import vstack
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img


model = load_model('G_model2/C3_256g_000042000.keras')
height, width = 256, 256


# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']
    # plot images row by row
    for i in range(len(images)-1):
        # define subplot
        pyplot.subplot(1, 2, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i], cmap="gray")
        # show title
        pyplot.title(titles[i])
    pyplot.show()


def preprocess_data(data):
    # load compressed arrays
    # unpack arrays
    X1, X2 = data[0], data[1]
    # scale from [0,255] to [-1,1]
    X1 = ((X1 - 127.5) / 127.5)
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


src_image = load_img(r'dataset/train_1/s1/ROIs1970_fall_s1_11_p61.png',
                     target_size=(height, width))
src_image = img_to_array(src_image)
src_image = np.expand_dims(src_image, axis=0)

tar_image = load_img(r'D:/v_2/agri/s1/ROIs1970_fall_s1_133_p1006.png',
                     target_size=(height, width))
tar_image = img_to_array(tar_image)
tar_image = np.expand_dims(tar_image, axis=0)

src_image, tar_image = preprocess_data([src_image, tar_image])

print(src_image.shape)
# generate image from source
gen_image = model.predict(src_image)

# plot all three images
plot_images(src_image, gen_image, tar_image)
