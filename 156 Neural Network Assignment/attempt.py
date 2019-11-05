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
from keras.utils.np_utils import to_categorical   


#load data
train = np.load('train_data.npy')
labels = np.load('train_label.npy')
test = np.load('test_data.npy') #test at very end

X = train
y = labels

X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=.15, random_state=42)

images_train = X_train
lables_train = y_train
images_test = X_test
labels_test = y_test

X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0, 2, 3, 1)

X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0, 2, 3, 1)

test = test.reshape((len(test), 3, 32, 32)).transpose(0, 2, 3, 1)
input_shape = (32, 32, 3)

# Making sure that the values are float so that we can get decimal points after division
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
test = test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_test /= 255
test /= 255

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator( rotation_range=90,
                 width_shift_range=0.1, height_shift_range=0.1,
                 horizontal_flip=True)
datagen.fit(X_train)

# train_datagen = ImageDataGenerator(
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True)   # flip images horizontally

# validation_datagen = ImageDataGenerator()

# train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
# validation_generator = validation_datagen.flow(X_train, y_train, batch_size=32)


from keras.optimizers import Adam
from keras.applications import ResNet50

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3)))
# Batch normalization layer added here
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
# Batch normalization layer added here
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
adam = Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, decay=0.0)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# history = model.fit_generator(train_generator,    
#                     validation_data=validation_generator,
#                     validation_steps=len(X_train) / 32,
#                     steps_per_epoch=len(X_train) / 32,
#                     epochs=150,
#                     verbose=2)

#plotLosses(history)
mc = keras.callbacks.ModelCheckpoint('newmodel{epoch}.h5', save_weights_only=False, period=50)
model.fit(x=X_train, y=y_train, callbacks=[mc], epochs=150)

score = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print(model.metrics_names)
print(score)
#model.save("newmodel100.h5")

