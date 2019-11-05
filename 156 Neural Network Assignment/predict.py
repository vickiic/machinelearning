#!/usr/bin/env python3
import pandas as pd
import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import numpy as np

#load data
train = np.load('train_data.npy')
labels = np.load('train_label.npy')
test = np.load('test_data.npy') #test at very end
test = test.reshape((len(test), 3, 32, 32)).transpose(0, 2, 3, 1)
input_shape = (32, 32, 3)
# Making sure that the values are float so that we can get decimal points after division
test = test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.

test /= 255

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam

model = keras.models.load_model("newmodel50.h5")
predictions = model.predict_classes(test)
#print(predictions.shape)
#np.savetxt("prediction.csv", predictions, delimiter=",")
#print(predictions[0]) #all field names of numpy file
#print(check.shape)
#np.savetxt("check.csv", check, delimiter=",")
#print(check[0])
np.save("prediction", predictions)