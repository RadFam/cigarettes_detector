import os
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

filename = 'D:\\DA_Projects\\CV_Projects\\Cigarettes\\cig_butts\\real_test\\0002.JPG'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
img = img / 255.0
img = np.array(img)
img = np.expand_dims(img, 0)

model = tf.keras.models.load_model('Final_model.h5', compile=False)
result = model.predict(img)
result = result * 512
print(result)