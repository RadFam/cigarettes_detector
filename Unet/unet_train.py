import os
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

BATCH = 5
EPOCHS = 15

# os.environ['PATH'] = os.environ['PATH']+';' + r"D:\\Distribs\\Graphviz\\bin"
# dot_img_file = 'model_1.png'

# def get_model(img_size, num_classes):
#     inputs = Input(shape=img_size + (3,))

#     ### [First half of the network: downsampling inputs] ###

#     # Entry block
#     x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)

#     previous_block_activation = x  # Set aside residual

#     # Blocks 1, 2, 3 are identical apart from the feature depth.
#     for filters in [64, 128, 256]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

#         # Project residual
#         residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual

#     ### [Second half of the network: upsampling inputs] ###

#     for filters in [256, 128, 64, 32]:
#         x = layers.Activation("relu")(x)
#         x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.Activation("relu")(x)
#         x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.UpSampling2D(2)(x)

#         # Project residual
#         residual = layers.UpSampling2D(2)(previous_block_activation)
#         residual = layers.Conv2D(filters, 1, padding="same")(residual)
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual

#     # Add a per-pixel classification layer
#     outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

#     # Define the model
#     model = Model(inputs, outputs)
#     return model


# # Build model
# model = get_model((224, 224), 2)



# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

def double_conv_block(x, num_filters):

    x = Conv2D(num_filters, kernel_size=3, stride=1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filters, kernel_size=3, stride=1, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    return x

def full_layer_block(x, num_filetrs):

    x_1 = double_conv_block(x, num_filetrs)
    x_2 = MaxPooling2D(pool_size=(2,2))(x_1)

    return x_1, x_2

def upsample_block(x, y, num_filters):

    x = Conv2DTranspose(num_filters, kernel_size=3, strides=(2,2), padding="same", activation="relu")(x)
    x = concatenate([x, y])
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

    output = Conv2D(1, kernel_size=(3,3), stride=1, padding="same", activation="softmax")(z_9)

    full_model = Model(x, output, name="UNet")

    return full_model

def read_train(data_train):
    cnt = 0
    while True:
        image_data = []
        mask_data = []
        for k in range(BATCH):
            if k == len(data_train):
                k = 0
            image_filename = data_train.iloc[cnt, data_train.columns.get_loc("filename")]
            

def read_valid(data_valid):
    pass

data_train = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\train.csv")
data_valid = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\valid.csv")