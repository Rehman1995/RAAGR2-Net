import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "-1";

import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, concatenate
#import tensorflow_addons as tfa
from tensorflow.keras.backend import squeeze
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import cv2
import matplotlib.pyplot as plt

PATH    = os.getcwd()

def dice_coef(y_true, y_pred, smooth=1):
    

    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=[0])
    return dice
'''
In case of binary IoU both functions below work exactly the same 
    i.e. the number of op_channel == 1
'''
def mean_iou(y_true, y_pred, smooth=1):
    
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou

def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss

def Weighted_BCEnDice_loss(y_true, y_pred):
    
    
    
   
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss

#%%
class SurvedModel:

    def __init__(self):
        '''
        Model should be loaded on memory here.  
        '''
        self.cl = { 'Weighted_BCEnDice_loss' : Weighted_BCEnDice_loss,
                      'mean_iou': mean_iou,
                      'dice_coef': dice_coef}
        self.model = tf.keras.models.load_model(filepath=PATH + '/DLU_Net_v20.h5',
                                                custom_objects=self.cl, compile=True) 
    
        
    def predict (self, img):
        '''
        Preprocessing & inference & postprocessing part.
        # img;attribute = {shape:[H, W, 3],  type : ndarray}
        # or in brain scans case a [H, W, 4] ndarray
        # return;attribute = {shape : [H, W, 3], type : ndarray}
        
        # return your_postprocessing(self.your_model(your_preprocessing(img)))
        '''
        print('start from inside')
        alpha = 0.5
        beta = (1.0 - alpha)
        
        img = img[np.newaxis, :, :, :]
        preds_train_loaded = np.squeeze(self.model.predict(img, verbose=1))
        preds_train = (preds_train_loaded > 0.2).astype(np.uint8)
        
        ip = (img[0,:,:, 0] * 255).astype(np.uint8)
        ip = cv2.merge((ip, ip, ip)) # make 3 channels for merging
        preds = preds_train[:,:,2:5]
        preds = (preds * 255).astype(np.uint8)
        
        o_pred = cv2.addWeighted(ip, alpha, preds, beta, 0.0)
        print('sent op')
        return o_pred