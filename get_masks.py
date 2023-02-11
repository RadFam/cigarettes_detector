import json
import os
import cv2
import numpy as np

def create_mask_image(filename, segmentation, img_size):
    mask = np.zeros((img_size[0], img_size[1], 3))
    for seg in segmentation:
        coords = np.array(seg)
        coords = np.reshape(coords, (int(coords.shape[0]/2), 2))
        #coords = np.reshape(coords, (-1, 1, 2))
        coords = coords.astype('uint64')
        cv2.drawContours(mask, [coords], -1, (255, 255, 255), -1)

    cv2.imwrite(filename=filename, img=mask)

# 1.
# work on train masks
upperfolder = os.path.dirname(os.path.dirname(__file__))
filename = upperfolder + "\\cig_butts\\train\\coco_annotations.json"

with open(filename, 'r') as jsn_file:
    jsn_data = json.load(jsn_file)

print(jsn_data.keys())
img_data = [(upperfolder + "\\cig_butts\\train\\images" + k['file_name'], (k['width'], k['height']), k['id']) for k in jsn_data['images']]
img_masks = [(k['segmentation'], k['image_id']) for k in jsn_data['annotations']]

# create folder
mask_folder = upperfolder + "\\cig_butts\\train\\masks"
if not os.path.exists(mask_folder):
    os.mkdir(mask_folder)

for data, mask in zip(img_data, img_masks):
    if mask[1] == data[2]:
        filename = mask_folder + "\\mask_" + str(data[2]).zfill(6) + ".png"
        create_mask_image(filename, mask[0], data[1])

# 2.
# work on validation masks
upperfolder = os.path.dirname(os.path.dirname(__file__))
filename = upperfolder + "\\cig_butts\\val\\coco_annotations.json"

with open(filename, 'r') as jsn_file:
    jsn_data = json.load(jsn_file)

print(jsn_data.keys())
img_data = [(upperfolder + "\\cig_butts\\val\\images" + k['file_name'], (k['width'], k['height']), k['id']) for k in jsn_data['images']]
img_masks = [(k['segmentation'], k['image_id']) for k in jsn_data['annotations']]

# create folder
mask_folder = upperfolder + "\\cig_butts\\val\\masks"
if not os.path.exists(mask_folder):
    os.mkdir(mask_folder)

for data, mask in zip(img_data, img_masks):
    if mask[1] == data[2]:
        filename = mask_folder + "\\mask_" + str(data[2]).zfill(6) + ".png"
        create_mask_image(filename, mask[0], data[1])