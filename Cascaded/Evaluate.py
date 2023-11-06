"""
===================================================================================================================
Package
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt

import keras
from keras import utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print()

from Model import cascaded_unet
from Loss import *
from Helper import *

model_path = "/home/b09508004/Brain/Result/Cascaded.h5"


"""
===================================================================================================================
Training Data Preparing
"""
batch = 32

train_generator, train_step = datagen("/home/b09508004/Data/Training/images/",
                                      "/home/b09508004/Data/Training/labels/",
                                      batch)

val_generator, val_step = datagen("/home/b09508004/Data/Training/images/",
                                  "/home/b09508004/Data/Training/labels/",
                                  batch)

test_generator, test_step = datagen("/home/b09508004/Data/Testing/images/",
                                    "/home/b09508004/Data/Testing/labels/",
                                    batch)

print()


"""
===================================================================================================================
Load Model
"""
model = cascaded_unet(img_rows = 128, img_cols = 128, color_type = 1, num_class = 1, inference = False)
             
model.load_weights(model_path)

adam = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = adam, loss = focal_tversky_loss, metrics = [mean_iou, dice_coef])


"""
===================================================================================================================
Evaluate Model
"""
(loss, iou, dice) = model.evaluate(train_generator, steps = train_step)

loss = [np.round(loss, 6)]
iou = [np.round(iou, 6)]
dice = [np.round(dice, 6)]

print('Training:')
print('Loss:', loss)
print('Mean IoU:', iou)
print('Dice Coef:', dice)
print()

(loss, iou, dice) = model.evaluate(val_generator, steps = val_step)

loss = [np.round(loss, 6)]
iou = [np.round(iou, 6)]
dice = [np.round(dice, 6)]

print('Validation:')
print('Loss:', loss)
print('Mean IoU:', iou)
print('Dice Coef:', dice)
print()

(loss, iou, dice) = model.evaluate(test_generator, steps = test_step)

loss = [np.round(loss, 6)]
iou = [np.round(iou, 6)]
dice = [np.round(dice, 6)]

print('Testing:')
print('Loss:', loss)
print('Mean IoU:', iou)
print('Dice Coef:', dice)
print()


"""
===================================================================================================================
Predict Mask
"""
img = utils.load_img("/home/b09508004/Data/Training/images/images/2PP6XTUA_45.bmp",
                     grayscale = True, target_size = (128, 128))

label = utils.load_img("/home/b09508004/Data/Training/labels/labels/2PP6XTUA_45.bmp", 
                        grayscale = True, target_size = (128, 128))

answer = utils.img_to_array(img)
answer = np.expand_dims(answer, axis = 0)
answer /= 255.
answer = model.predict(answer)
answer = answer[2]
answer = answer.reshape(128, 128, 1)
answer = utils.array_to_img(answer)

plt.subplot(1, 3, 1)
plt.imshow(img, plt.cm.gray)
plt.subplot(1, 3, 2)
plt.imshow(answer, plt.cm.gray)
plt.subplot(1, 3, 3)
plt.imshow(label, plt.cm.gray)
plt.show()