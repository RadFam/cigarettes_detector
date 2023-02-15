import json
import os
import cv2
import numpy as np
import pandas as pd

# 1.
# work on train images
upperfolder = os.path.dirname(os.path.dirname(__file__))
filename = upperfolder + "\\cig_butts\\train\\coco_annotations.json"

with open(filename, 'r') as jsn_file:
    jsn_data = json.load(jsn_file)

img_data = [(upperfolder + "\\cig_butts\\train\\images\\" + k['file_name'], (k['width'], k['height']), k['id']) for k in jsn_data['images']]
img_bbox = [(k['bbox'], k['image_id']) for k in jsn_data['annotations']]

full_data = []
for img, bbox in zip(img_data, img_bbox):
    if img[2] == bbox[1]:
        full_data.append([img[0], img[1][0], img[1][1], bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]])

# save full_data as bin?
columns = ['filename', 'width', 'height', 'x_1', 'y_1', 'x_2', 'y_2']

data = pd.DataFrame(columns=columns, data=full_data)
data.to_csv('train.csv', index=False)

# 2.
# work on validation images
filename = upperfolder + "\\cig_butts\\val\\coco_annotations.json"

with open(filename, 'r') as jsn_file:
    jsn_data = json.load(jsn_file)

img_data = [(upperfolder + "\\cig_butts\\val\\images\\" + k['file_name'], (k['width'], k['height']), k['id']) for k in jsn_data['images']]
img_bbox = [(k['bbox'], k['image_id']) for k in jsn_data['annotations']]

full_data = []
for img, bbox in zip(img_data, img_bbox):
    if img[2] == bbox[1]:
        full_data.append([img[0], img[1][0], img[1][1], bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]])

# save full_data as bin?
columns = ['filename', 'width', 'height', 'x_1', 'y_1', 'x_2', 'y_2']

data = pd.DataFrame(columns=columns, data=full_data)
data.to_csv('valid.csv', index=False)