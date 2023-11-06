"""
===================================================================================================================
Package
"""
import warnings
warnings.filterwarnings('ignore')

import openpyxl

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
print()
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
print()

from Model import cascaded_unet
from Loss import *
from Helper import *

filepath = "/home/b09508004/Brain/Result/"
result_path = "/home/b09508004/Brain/Cascaded/Result/Result.xlsx"


"""
===================================================================================================================
Data Preparing
"""
batch = 128
epoch = 25
print('Batch Size:', batch)
print('Epoch:', epoch)

train_generator, train_step = datagen("/home/b09508004/Data/Training/images/",
                                      "/home/b09508004/Data/Training/labels/",
                                      batch)

val_generator, val_step = datagen("/home/b09508004/Data/Training/images/",
                                  "/home/b09508004/Data/Training/labels/",
                                  batch)

print()


"""
===================================================================================================================
Build Model
"""
model = cascaded_unet(img_rows = 128, img_cols = 128, color_type = 1, num_class = 1)

if os.path.exists(filepath + 'Cascaded_Final.h5'):
    model.load_weights(filepath + 'Cascaded_Final.h5')
    print('Load Weight')
    print()


"""
===================================================================================================================
Complie Model + Callbacks
"""
adam = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = adam, loss = focal_tversky_loss, metrics = [mean_iou, dice_coef])

model_path = filepath + 'Cascaded.h5'
checkpoint = ModelCheckpoint(model_path, monitor = 'val_dice_coef', mode = 'max', verbose = 1, 
                              save_weights_only = True, save_best_only = True)

def scheduler(epoch, lr):
  
  if epoch < 5:

    warmup_percent = (epoch + 1) / 5

    return 1e-4 * warmup_percent
  
  else:

    return lr ** 1.01

lr_scaheduler = LearningRateScheduler(scheduler, verbose = 1)


"""
===================================================================================================================
Fit Model
"""
history = model.fit(train_generator,
                    steps_per_epoch = train_step,
                    epochs = epoch, 
                    verbose = 1, 
                    shuffle = True,
                    validation_data = val_generator,
                    validation_steps = val_step,
                    callbacks = [checkpoint, lr_scaheduler])
                    
model.save(filepath + 'Cascaded_Final.h5')


"""
===================================================================================================================
Loss Function
"""
get_loss(history, result_path)