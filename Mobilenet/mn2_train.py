import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStoppping
from tensorflow.kears.callbacks import ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

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

callback = EarlyStoppping(monitor='loss', patience=1, min_delta=1e-02, )

final_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


model.summary()