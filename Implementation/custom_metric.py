import tensorflow as tf 
import numpy as np
#if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0':
#    from keras.layers.merge import concatenate
#    from keras.layers import Activation
#    import keras.backend as K
#    from keras.backend import squeeze
#if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.3.0' or tf.__version__ == '2.5.0':
import keras.backend as K
from keras.layers import Activation, concatenate
#import tensorflow_addons as tfa
from keras.backend import squeeze 



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
