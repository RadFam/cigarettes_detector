import os
import tensorflow as tf
import pandas as pd
import cv2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStoppping
from tensorflow.kears.callbacks import ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['PATH'] = os.environ['PATH']+';' + r"D:\\Distribs\\Graphviz\\bin"

#model = MobileNetV2(input_shape=None, alpha=1, include_top=True, weights="imagenet", input_tensor=None, classes=1000, classifier_activation="softmax")
#model = MobileNetV2(input_shape=None, alpha=1, include_top=False, weights="imagenet", input_tensor=None)
mn = MobileNetV2(input_shape=None, alpha=1, include_top=False, weights="imagenet", input_tensor=None)

#dot_img_file = 'model_1.png'
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# Add new head
model = mn.output
model = Dense(128, activation='relu')(model)
model = Dense(64, activation='relu')(model)
model = Dense(32, activation='relu')(model)
model = Dense(4, activation='sigmoid')(model)

final_model = Model(inputs=mn.input, outputs=model)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
loss = MeanSquaredError()
metrics = [MeanSquaredError()]

callback = EarlyStoppping(monitor='loss', patience=1, min_delta=1e-02)
checkpoint = ModelCheckpoint(filepath='\\train_logs', save_weights_only=True, monitor='val_accuracy', save_best_only=True, model='min')

final_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

batch = 5
epochs = 5

# load train and valid datasets
data_train = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\train.csv")
data_valid = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + "\\valid.csv")

train_targets = data_train[['x_1', 'y_1', 'x_2', 'y_2']]
train_features = cv2.imread(data_train['filename'])
train_features = cv2.cvtColor(train_features, cv2.COLOR_BGR2RGB)

valid_targets = data_valid[['x_1', 'y_1', 'x_2', 'y_2']]
valid_features = cv2.imread(data_valid['filename'])
valid_features = cv2.cvtColor(valid_features, cv2.COLOR_BGR2RGB)

train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_features, train_targets, batch_size=batch)
valid_generator = valid_datagen.flow(valid_features, valid_targets, batch_size=batch)

history = final_model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=batch, callbacks=[callback], checkpoints=[checkpoint], verbose=1)


model.summary()
