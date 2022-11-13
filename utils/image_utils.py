import tensorflow as tf
from tensorflow import keras
from .image_processing import rescale
def unprocess_drc_preprocessing(x, drc_preproccesing_type='default'):
    if drc_preproccesing_type=='default':
        x = tf.multiply(x, 0.5)
        x = tf.add(x, 0.5)
    else:
        error
    return x

def rggb_to_rgb(x):
    return tf.stack([x[...,0],
                   0.5*(x[...,1]+x[...,2]),
                   x[...,3]],axis=-1)

def undo_rggb_channel_gains(x, mean_red_gain=2.25,mean_blue_gain=1.7, mean_rgb_gain=1.25):
    #source for the mena gains:
    # rgb_gain = 1.0 / tf.random.normal((), mean=0.8, stddev=0.1)
    # red_gain = tf.random.uniform((), 1.9, 2.4)
    # blue_gain = tf.random.uniform((), 1.5, 1.9)

    gains = tf.stack([1.0 * mean_red_gain, 1.0, 1.0 * mean_blue_gain]) * mean_rgb_gain
    #prepare to broadcast with a  flexible number of broadcast dimensions
    # gain_shape = [1 for _ in x.shape.as_list()]
    # gain_shape[-1]=3
    # gains = gains.reshape(gain_shape)
    return x * gains


#todo merge these two functions:
def upsample_and_reprocess(image,upsample_factor=4, preprocessing=None):
    image = unprocess_drc_preprocessing(image)
    image = keras.layers.UpSampling2D(size=(upsample_factor,upsample_factor))(image)
    image = rescale(image, preprocessing=preprocessing)
    return image

def upsample_rggb(x, upsample_factor=8, algo='vanilla',preprocessing='keras_resnet50'):
    x = rggb_to_rgb(x)
    x = unprocess_drc_preprocessing(x)
    x = undo_rggb_channel_gains(x)
    x = keras.layers.UpSampling2D(size=(upsample_factor,upsample_factor))(x)
    x = rescale(x, preprocessing=preprocessing)
    return x

