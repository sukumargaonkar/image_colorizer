# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 23:51:11 2018

@author: Karra's
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
import cv2

import matplotlib.pyplot as plt
import os


INPUT_IMAGE_SRC = 'D:/AI_4/train/urban/sam.jpeg'
display(Image(INPUT_IMAGE_SRC, width=225))

img = cv2.imread(INPUT_IMAGE_SRC)
plt.imshow(img)
           
image = img_to_array(load_img(INPUT_IMAGE_SRC, target_size=(200,200))) / 255
lab_image = rgb2lab(image)
lab_image.shape

lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]


# The input will be the black and white layer
X = lab_image_norm[:,:,0]
# The outpts will be the ab channels
Y = lab_image_norm[:,:,1:]

# The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
# (samples, rows, cols, channels), so we need to do some reshaping
X = X.reshape(1, X.shape[0], X.shape[1], 1)
Y = Y.reshape(1, Y.shape[0], Y.shape[1], 2)

model = Sequential()
model.add(InputLayer(input_shape=(200, 200, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))

# Finish model
model.compile(optimizer='rmsprop', loss='mse')

model.fit(x=X, y=Y, batch_size=1, epochs=1000, verbose=0)

model.evaluate(X, Y, batch_size=1)

output = model.predict(X)
cur = np.zeros((200, 200, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]


cur = (cur * [100, 255, 255]) - [0, 128, 128]
rgb_image = lab2rgb(cur)
imshow(rgb_image)
