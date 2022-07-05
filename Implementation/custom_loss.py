
import tensorflow as tf
import numpy as np 
 
#if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0':
#    from keras.layers.merge import concatenate
#    from keras.layers import Activation
#    import keras.backend as K 
#if tf.__version__ == '2.2.0' or tf.__version__ == '2.1.0' or tf.__version__ == '2.3.0' or tf.__version__ == '2.5.0': 
import keras.backend as K
from keras.layers import Activation, concatenate
    #import tensorflow_addons as tfa

'''
In this on the fly lossses the ground truths are converted on they fly into the categorical type and only 
hybrid and tri brid losses are doing that if you wann use only one loss then convert them first
'''
#-------------------------------------------------------------Dice Loss Function-----------------------



def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight=1):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss

def Dice_coef(y_true, y_pred, weight=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def Dice_loss(y_true, y_pred):
    loss = 1 - Dice_coef(y_true, y_pred)
    return loss
        

def Weighted_BCEnDice_loss(y_true, y_pred):
    
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1),padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss
