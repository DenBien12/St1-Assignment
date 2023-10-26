import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


data_dir = "archive/traffic_light_data/train/"
traffic_light = ['back', 'green', 'yellow', 'red']
img_size = (20, 100)
data = []
for color in traffic_light:
    folder = os.path.join(data_dir, color)
    label = traffic_light.index(color)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, img_size)
        data.append([img_arr, label])

random.shuffle(data)

X = []
y = []
for features, labels in data:
    X.append(features)
    y.append(labels)
X = np.array(X)
y = np.array(y)
X =X/255
X.shape

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(128,input_shape = X.shape[1:],activation ="relu"))

# number of classes
model.add(Dense(4,activation ="softmax"))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X,y, epochs = 10, validation_split = 0.1)

model.summary()
model.save(r'model_hand.h5')

