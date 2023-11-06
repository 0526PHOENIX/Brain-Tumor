"""
===================================================================================================================
Package
"""
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from keras import backend as K


"""
===================================================================================================================
Mean IoU
"""
def IoU(y_true, y_pred):
    
    smooth = 1e-6
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    
    return IoU
    
    
def mean_iou(y_true, y_pred):

    prec = []
    
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast((y_pred > t), tf.float32)
        iou = IoU(y_true, y_pred_)
        prec.append(iou)    
    
    return K.mean(K.stack(prec), axis = 0)
    
    
def iou_loss(y_true, y_pred):

    return 1. - IoU(y_true, y_pred)
    

"""
===================================================================================================================
Dice loss
"""
def dice_coef(y_true, y_pred):

    smooth = 1e-6
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    return dice


def dice_loss(y_true, y_pred):

    return 1. - dice_coef(y_true, y_pred)
    

"""
===================================================================================================================
Focal loss
"""  
def focal_loss(y_true, y_pred):    
    
    alpha = 0.8
    gamma = 2.
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    bce_exp = K.exp(-bce)
    
    loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)
    
    return loss
    

"""
===================================================================================================================
Tversky loss
"""  
def tversky(y_true, y_pred):
    
    smooth = 1e-6
    alpha = 0.7
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    
    tv = (true_pos + smooth) / (true_pos + (alpha * false_neg) + ((1 - alpha) * false_pos) + smooth)
    
    return tv


def tversky_loss(y_true, y_pred):

    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):

    gamma = 0.75
    
    tv = tversky(y_true, y_pred)
    
    return K.pow((1 - tv), gamma)