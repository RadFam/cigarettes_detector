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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_data(data_train, batch_size):
    cnt = 0
    while True:
        image_data = []
        features_data = []
        for k in range(batch_size):
            if cnt == len(data_train):
                cnt = 0
            filename = data_train.iloc[cnt, data_train.columns.get_loc("filename")]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            img = img / 255.0
            image_data.append(img)

            values = (data_train.iloc[cnt, [3, 4, 5, 6]]).to_list()
            for i, v in enumerate(values):
                values[i] = v / 512.0
            features_data.append(values)

            cnt += 1
        image_data = np.array(image_data)
        features_data = np.array(features_data)
        yield [image_data, features_data]


def get_valid_data(data_valid, batch_size):
    cnt = 0
    while True:
        image_data = []
        features_data = []
        for k in range(batch_size):
            if cnt == len(data_valid):
                cnt = 0
            filename = data_valid.iloc[cnt, data_valid.columns.get_loc("filename")]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            img = img / 255.0
            image_data.append(img)

            values = (data_valid.iloc[cnt, [3, 4, 5, 6]]).to_list()
            for i, v in enumerate(values):
                values[i] = v / 512.0
            features_data.append(values)

            cnt += 1
        image_data = np.array(image_data)
        features_data = np.array(features_data)
        yield [image_data, features_data]

# os.environ['PATH'] = os.environ['PATH']+';' + r"D:\\Distribs\\Graphviz\\bin"

#mn = MobileNetV2(input_shape=None, alpha=1, include_top=True, weights="imagenet", input_tensor=None, classes=1000, classifier_activation="softmax")
#mn = MobileNetV2(input_shape=None, alpha=1, include_top=False, weights="imagenet", input_tensor=None)
mn = MobileNetV2(input_shape=(224, 224, 3), alpha=1, include_top=False, weights="imagenet", input_tensor=Input(shape=(224, 224, 3)))

#dot_img_file = 'model_1.png'
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

mn.summary()
mn.trainable = False

# Add new head
model = mn.output
model = Flatten()(model)
model = Dense(256, activation='relu')(model)
model = Dense(100, activation='relu')(model)
model = Dense(32, activation='relu')(model)
model = Dense(4, activation='sigmoid')(model)

final_model = Model(inputs=mn.input, outputs=model)

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
loss = MeanSquaredError()
metrics = [MeanSquaredError()]

callback = EarlyStopping(monitor='loss', patience=10, min_delta=1e-02, verbose=1)
checkpoint = ModelCheckpoint(filepath='\\train_logs\\ep{epoch:03d}.h5', save_weights_only=True, monitor='val_accuracy', save_best_only=True, period=3, verbose=1)
reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=1e-03, patience=1, verbose=1)

final_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

batch = 5
epochs_1 = 10
epochs_2 = 10

# load train and valid datasets
data_train = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\train.csv")
data_valid = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\valid.csv")

# train_targets = data_train[['x_1', 'y_1', 'x_2', 'y_2']]
# train_features = []
# for index, row in data_train.iterrows():
#     img = cv2.imread(row['filename'])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     train_features.append(img)

# valid_targets = data_valid[['x_1', 'y_1', 'x_2', 'y_2']]
# valid_features = []
# for index, row in data_valid.iterrows():
#     img = cv2.imread(row['filename'])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     valid_features.append(img)

# train_datagen = ImageDataGenerator()
# valid_datagen = ImageDataGenerator()

# train_generator = train_datagen.flow(train_features, train_targets, batch_size=batch)
# valid_generator = valid_datagen.flow(valid_features, valid_targets, batch_size=batch)

# history = final_model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=batch, callbacks=[callback], checkpoints=[checkpoint], verbose=1)
mn.trainable = False
history = final_model.fit_generator(get_train_data(data_train, batch),
                                     steps_per_epoch = len(data_train) // batch, 
                                     validation_data = get_valid_data(data_valid, batch),
                                     validation_steps = len(data_valid) // batch,
                                     epochs=epochs_1, callbacks = [callback, checkpoint])
mn.trainable = True
history = final_model.fit_generator(get_train_data(data_train, batch), 
                                    steps_per_epoch = len(data_train) // batch, 
                                    validation_data = get_valid_data(data_valid, batch), 
                                    validation_steps = len(data_valid) // batch,
                                    epochs=epochs_2, callbacks = [callback, checkpoint])
final_model.save_weights('Final_weight.h5')
final_model.save('Final_model.h5')


#model.summary()