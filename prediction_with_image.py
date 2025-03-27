import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential

from keras.utils import img_to_array
from keras.utils import load_img

import os


def prediction_with_image(img_string, img_height, img_width, class_names, model_name):
    #Load image
    img = load_img(img_string, target_size=(img_height, img_width))
    #Make prediction
    img_array = tf.expand_dims(img_to_array(img), 0)
    #The prediction will be a tuple of numbers. The largest number corresponds to the predicted label
    prediction = model_name.predict(img_array) 
    #Make plot
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    #Extract the real label of the potato leaf from the filename
    plt.xlabel(os.path.basename(img_string)[0:-4])
    plt.subplot(1,2,2)
    plt.bar(range(len(class_names)),prediction[0])
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    plt.ylim([0, 1])
    plt.xlabel('{} with a confidence of {:2.0f}%'.format(class_names[np.argmax(prediction)], 100*np.max(prediction)))  
    return plt.show()                       