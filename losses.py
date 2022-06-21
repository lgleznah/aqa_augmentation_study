# Custom loss functions for TensorFlow models
# Taken from code from ferrubio, in the private repository: https://github.com/ferrubio/AQA-framework

import tensorflow as tf
import tensorflow.keras.backend as K

# EMD loss
def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)