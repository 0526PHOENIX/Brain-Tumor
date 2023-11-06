"""
===================================================================================================================
Package
"""
import warnings
warnings.filterwarnings('ignore')
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

from Model import Xnet
from Loss import *
from Helper import *

model_path = "/home/b09508004/Brain/Result/Xnet.h5"


"""
===================================================================================================================
Data Preparing
"""
batch = 32

train_generator, train_step = datagen("/home/b09508004/Data/Training/images/",
                                      "/home/b09508004/Data/Training/labels/",
                                      batch)

val_generator, val_step = datagen("/home/b09508004/Data/Validation/images/",
                                  "/home/b09508004/Data/Validation/labels/",
                                  batch)

test_generator, test_step = datagen("/home/b09508004/Data/Testing/images/",
                                    "/home/b09508004/Data/Testing/labels/",
                                    batch)

print()


"""
===================================================================================================================
Set Model
"""
model = Xnet(backbone_name = 'densenet121',
             encoder_weights = 'imagenet',
             classes = 1,
             activation = 'sigmoid')
             
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
Mask
"""
img = utils.load_img("/home/b09508004/Data/Testing/images/images/YWNW6ARU_94.bmp", 
                     grayscale = False, target_size = (224, 224))
label = utils.load_img("/home/b09508004/Data/Testing/labels/labels/YWNW6ARU_94.bmp", 
                        grayscale = True, target_size = (224, 224))

answer = utils.img_to_array(img)
answer = np.expand_dims(answer, axis = 0)
answer /= 255.
answer = model.predict(answer)
answer = answer[0, :, :, :]
answer = utils.array_to_img(answer)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img, plt.cm.gray)
ax[1].imshow(answer, plt.cm.gray)
ax[2].imshow(label, plt.cm.gray)

ax[0].set_title('Image')
ax[1].set_title('Predict')
ax[2].set_title('Lable')

fig.tight_layout()
plt.show()