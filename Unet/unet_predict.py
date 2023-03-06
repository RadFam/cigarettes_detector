import tensorflow as tf
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

filename = 'D:\\DA_Projects\\CV_Projects\\Cigarettes\\cig_butts\\real_test\\0009.JPG'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
img = img / 255.0
img = np.array(img)
img = np.expand_dims(img, 0)

model = tf.keras.models.load_model('unet_model.h5', compile=False)
result = model.predict(img)

result_2 = create_mask(result)

plt.imshow(tf.keras.utils.array_to_img(result_2))
plt.show()

# result_2 = result_2.numpy()
# res_img = result_2 * 255
# res_img = np.reshape(res_img, (224, 224))
# res_img = res_img.astype('uint32')
# #res_img = cv2.resize(res_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

# cv2.imshow('res', res_img)
# cv2.waitKey(0)
a = 0