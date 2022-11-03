import tensorflow as tf
from tensorflow import keras

def upsample_rggb(x, upsample_factor=8, algo='vanilla',preprocessing='keras_resnet50'):
    x = tf.stack([x[...,0],
                   0.5*(x[...,1]+x[...,2]),
                   x[...,3]],axis=-1)
    x = keras.layers.UpSampling2D(size=(upsample_factor,upsample_factor))(x)
    if preprocessing == 'keras_resnet50':
        x = tf.cast(x * (256.), tf.float32)
        x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

