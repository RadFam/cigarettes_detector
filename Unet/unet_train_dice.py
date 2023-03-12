import os
import pandas as pd
import cv2
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


BATCH = 10
EPOCHS = 5

def double_conv_block(x, num_filters):

    x = Conv2D(num_filters, kernel_size=3, strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filters, kernel_size=3, strides=(1,1), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    return x

def full_layer_block(x, num_filetrs):

    x_1 = double_conv_block(x, num_filetrs)
    x_2 = MaxPooling2D(pool_size=(2,2))(x_1)
    x_2 = Dropout(0.3)(x_2)

    return x_1, x_2

def upsample_block(x, y, num_filters):

    x = Conv2DTranspose(num_filters, kernel_size=3, strides=(2,2), padding="same", activation="relu")(x)
    x = concatenate([x, y])
    x = Dropout(0.3)(x)
    x = double_conv_block(x, num_filters)

    return x

def create_unet(init_shape=(224, 224, 3)):

    x = Input(shape = init_shape)

    x_1, y_1 = full_layer_block(x, 64)

    x_2, y_2 = full_layer_block(y_1, 128)

    x_3, y_3 = full_layer_block(y_2, 256)

    x_4, y_4 = full_layer_block(y_3, 512)

    bottleneck = double_conv_block(y_4, 1024)

    z_6 = upsample_block(bottleneck, x_4, 512)

    z_7 = upsample_block(z_6, x_3, 256)

    z_8 = upsample_block(z_7, x_2, 128)

    z_9 = upsample_block(z_8, x_1, 64)

    output = Conv2D(1, kernel_size=(1,1), padding="same", activation="softmax", kernel_initializer='he_normal')(z_9)

    full_model = Model(x, output, name="UNet")

    return full_model

def read_train(data_train):
    cnt = 0
    while True:
        image_data = []
        mask_data = []
        for k in range(BATCH):
            if cnt == len(data_train):
                cnt = 0

            image_filename = data_train.iloc[cnt, data_train.columns.get_loc("image_path")]
            mask_filename = data_train.iloc[cnt, data_train.columns.get_loc("mask_path")]

            img = cv2.imread(image_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            img = img / 255.0
            image_data.append(img)

            mask = cv2.imread(mask_filename)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            mask = mask / 255
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype('uint32')
            mask_data.append(mask)

            cnt += 1

        image_data = np.array(image_data)
        mask_data = np.array(mask_data)

        yield [image_data, mask_data]
            

def read_valid(data_valid):
    cnt = 0
    while True:
        image_data = []
        mask_data = []
        for k in range(BATCH):
            if cnt == len(data_valid):
                cnt = 0

            image_filename = data_valid.iloc[cnt, data_valid.columns.get_loc("image_path")]
            mask_filename = data_valid.iloc[cnt, data_valid.columns.get_loc("mask_path")]

            img = cv2.imread(image_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            img = img / 255.0
            image_data.append(img)

            mask = cv2.imread(mask_filename)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            mask = mask / 255
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype('uint32')
            mask_data.append(mask)

            cnt += 1

        image_data = np.array(image_data)
        mask_data = np.array(mask_data)

        yield [image_data, mask_data]


def dice_lost(y_true, y_pred, smooth=1e-6):
    y_pred = K.cast(y_pred, dtype=tf.float32)
    y_true = K.cast(y_true, dtype=tf.float32)

    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_pred_f * y_true_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

data_train = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\train_image_mask.csv")
data_valid = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\valid_image_mask.csv")

full_model = create_unet()

#tf.keras.utils.plot_model(full_model, to_file=dot_img_file, show_shapes=True)

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
loss = SparseCategoricalCrossentropy(from_logits=True)
metrics = Accuracy()

full_model.compile(optimizer=optimizer, loss=dice_lost, metrics="sparse_categorical_accuracy")
full_model.fit_generator(read_train(data_train), 
                                   steps_per_epoch=len(data_train) // BATCH,
                                   validation_data = read_valid(data_valid),
                                   validation_steps=len(data_valid) // BATCH,
                                   epochs=EPOCHS,
                                )
                                

full_model.save('unet_model.h5')
