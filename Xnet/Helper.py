"""
===================================================================================================================
Package
"""
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator


"""
===================================================================================================================
Data Preparing
"""
def datagen(image_path, label_path, batch):

    image_datagen = ImageDataGenerator(rescale = 1. / 255)
    label_datagen = ImageDataGenerator(rescale = 1. / 255)

    image_generator = image_datagen.flow_from_directory(image_path, target_size = (224, 224),
                                                        class_mode = None, color_mode = 'rgb',
                                                        batch_size = batch, seed = 123)

    label_generator = label_datagen.flow_from_directory(label_path, target_size = (224, 224),
                                                        class_mode = None, color_mode = 'grayscale',
                                                        batch_size = batch, seed = 123)

    size = len(image_generator)

    data_generator = zip(image_generator, label_generator)

    return data_generator, size


"""
===================================================================================================================
Loss Function
"""
def get_loss(history, filepath):

    """
    ===============================================================================================================
    Get Loss Value
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    iou = history.history['mean_iou']
    val_iou = history.history['val_mean_iou']
    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']


    """
    ===============================================================================================================
    Save Loss Value
    """
    new = pd.DataFrame()
    new['loss'] = loss
    new['val_loss'] = val_loss
    new['Dice'] = dice
    new['Val_Dice'] = val_dice
    new['IoU'] = iou
    new['Val_IoU'] = val_iou
    
    old = pd.read_excel(filepath)
    combined = old.append(new)
    combined.to_excel(filepath, index = False)


    """
    ===============================================================================================================
    Draw Loss Value
    """
    epochs = range(1, len(iou) + 1)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epochs, iou, 'b', label = 'Training')
    ax[0].plot(epochs, val_iou, 'r', label = 'Validation')
    ax[1].plot(epochs, dice, 'b', label = 'Training')
    ax[1].plot(epochs, val_dice, 'r', label = 'Validation')

    ax[0].set_title('Mean IoU')
    ax[1].set_title('Dice Coeff')

    fig.tight_layout()
    plt.show()