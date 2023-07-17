# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# The code was taken from:
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
#
# I've renamed the file and modify the code to suit my own need.
# JK Jung, <jkjung13@gmail.com>
# I've further modified code for my own needs, Alexander Rivkind


"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import control_flow_ops
import utils.unprocess as unprocess
from tensorflow import keras


def rescale(x,enforce_def=False,preprocessing='default',**kwargs):

    if enforce_def:
        preprocessing = 'default'

    if preprocessing == 'keras_resnet50':
        x = tf.cast(x * (256.), tf.float32)
        x = tf.keras.applications.resnet50.preprocess_input(x)
    elif preprocessing == 'keras_mobilenet_v2':
        x = tf.cast(x * (256.), tf.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        print(
            'debug ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssmobilenet')
    elif preprocessing == 'VGG16':
        x = tf.cast(x * (256.), tf.float32)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        print(
            'debug ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssVGG16')
    elif preprocessing == 'VGG19':
        x = tf.cast(x * (256.), tf.float32)
        x = tf.keras.applications.vgg19.preprocess_input(x)
        print(
            'debug ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssVGG16')
    elif preprocessing == 'alex_net':
        channels = tf.unstack(x, axis=-1)
        x = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        x = x - tf.reshape(tf.reduce_mean(x, axis=[-2, -3]), [1, 1, 3])
        x = tf.cast(x * (256.), tf.float32)
        print(
            'debug ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssAlexNet')
    elif preprocessing == 'alex_net2':
        raise NotImplementedError
        # channels = tf.unstack(x, axis=-1)
        # x = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # x = x  - tf.reshape(tf.reduce_mean(x,axis=[-2,-3]),[1,1,3])
        # x = tf.cast(x * (256.), tf.float32)
        # print(
        #     'debug ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssAlexNet')
    elif preprocessing == 'default':
        x = tf.subtract(x, 0.5)
        x = tf.multiply(x, 2.0)
    elif preprocessing == 'identity':
        pass
    else:
        raise NotImplementedError

    return x


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.cast(height,tf.float32)
  width = tf.cast(width,tf.float32)
  smallest_side = tf.cast(smallest_side,tf.float32)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.cast(tf.math.rint(height * scale),tf.int32)
  new_width = tf.cast(tf.math.rint(width * scale),tf.int32)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize(image, [new_height, new_width],method='bilinear')
                                         #  align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def _crop(image, offset_height, offset_width, crop_height, crop_width, resize=None):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
      # if resize is None:
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
      # else:
      #   cropped_shape = tf.stack([resize[0], resize[1], original_shape[2]])
  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.cast(tf.stack([offset_height, offset_width, 0]),tf.int32)

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)

  if resize is None:
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
  else:
    image = tf.image.resize(image, resize , method='bilinear')
    cropped_shape = tf.stack([resize[0], resize[1], original_shape[2]])

  return tf.reshape(image, cropped_shape)


def _central_crop(image_list, crop_height, crop_width, **kwargs):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    if kwargs['central_squeeze_and_pad_factor'] > 0:
        image = _central_squeeze_and_pad(image, kwargs['central_squeeze_and_pad_factor'])

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

def organize_padding(x0,x1):
    delta = x0 - x1
    delta = delta//2 +1
    return [delta, delta]

def _central_squeeze_and_pad(image, squeeze_factor):
    x = image
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    squeezed_height = tf.cast(squeeze_factor*tf.cast(image_height,tf.float32),tf.int32)
    squeezed_width = tf.cast(squeeze_factor*tf.cast(image_width,tf.float32),tf.int32)
    # size = [image_height, image_width]
    # imresize = [squeezed_height, squeezed_width]
    # neutral_color = tf.reduce_mean(image,axis=[0,1]) # [0, 0, 0, 0]  # pixel from the image surrounding
    neutral_color = tf.reduce_mean(image,axis=[0,1]) # [0, 0, 0, 0]  # pixel from the image surrounding
    #2:
    x = keras.layers.Resizing(squeezed_height,
                              squeezed_width,
                              interpolation='bilinear',
                              crop_to_aspect_ratio=False, )(x)

    height_padding = organize_padding(image_height, squeezed_height)
    width_padding = organize_padding(image_width, squeezed_width)
    # x = tf.pad(
    #     x, [height_padding, width_padding, [0,0]], mode='CONSTANT', constant_values=neutral_color, name=None
    #     )

    x = tf.stack( [tf.pad(
         x[:,:,ii], [height_padding, width_padding], mode='CONSTANT', constant_values=neutral_color[ii], name=None
         ) for ii in range(3)],
    axis=2)

    return x

def _central_crop_with_offsets(image, high_res, low_res, offsets, n_steps,**kwargs):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  low_res_frames = []
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  if kwargs['central_squeeze_and_pad_factor'] > 0:
      image = _central_squeeze_and_pad(image, kwargs['central_squeeze_and_pad_factor'])

  offset_height0 = (image_height - high_res) // 2
  offset_width0 = (image_width - high_res) // 2

  high_res_image = _crop(image, offset_height0, offset_width0,
                       high_res, high_res)

  # for offset in offsets:
  #   frame = _crop(image, offset_height0 + offset[0], offset_width0 + offset[1],
  #           high_res, high_res,resize=[low_res,low_res])
  #   low_res_frames.append(frame)
  for step in range(n_steps):
        frame = _crop(image, offset_height0 + offsets[step,0], offset_width0 + offsets[step,1],
                      high_res, high_res, resize=[low_res, low_res])
        low_res_frames.append(frame)

  low_res_frames = tf.stack(low_res_frames)
  return low_res_frames, high_res_image


def _central_crop_with_offsets_rggb(image, high_res, low_res, offsets, n_steps, unprocess_high_res=False,**kwargs):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  kwargs.setdefault('rggb_teacher', False)

  low_res_frames = []
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  offset_height0 = (image_height - high_res) // 2
  offset_width0 = (image_width - high_res) // 2

  if kwargs['rggb_teacher']:
      #image is the raw image used for obtaining low resolution crops
      #high_res_image is the cropped version which should match the low resolution crops
      image, _ = unprocess.unprocess(image, do_mosaic=False, **kwargs)

      high_res_image = _crop(image, offset_height0, offset_width0,
                           high_res, high_res)

      #prior to mosaic step the image is upsampled:
      high_res_image = tf.image.resize(high_res_image, [high_res * 2, high_res * 2], method='bilinear')
      high_res_image = unprocess.mosaic(high_res_image)

      #another, more lossy option is to rescale after mosaicing
      # high_res_image = tf.image.resize(high_res_image, [high_res, high_res], method='bilinear')

      print('debug ------------------------------100',kwargs['rggb_teacher'])
  else:
      high_res_image = _crop(image, offset_height0, offset_width0,
                           high_res, high_res)
      image, _ = unprocess.unprocess(image, do_mosaic=False, **kwargs)
      print('debug ------------------------------200')

  for step in range(n_steps):
        frame = _crop(image, offset_height0 + offsets[step,0], offset_width0 + offsets[step,1],
                      high_res, high_res, resize=[low_res, low_res])
        if unprocess_high_res:
            #this is the default setting where unprocessing is done only once, for the high resolution image
            #and the crops are performed subsequently so that all the crpped images share the same random channel gains
            #the noise however is generated for each crop separately in both cases
            frame = unprocess.mosaic(frame)
        else:
            #we keep this option for backward compatibility
            frame,_ = unprocess.unprocess(frame, do_mosaic=True, **kwargs)
            print('debug ------------------------------300')

        shot_noise, read_noise = unprocess.random_noise_levels()
        frame = unprocess.add_noise(frame, shot_noise, read_noise)
        low_res_frames.append(frame)

  low_res_frames = tf.stack(low_res_frames)
  return low_res_frames, high_res_image


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  if True:
      #with tf.name_scope(scope):#, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  if True:
  # with tf.name_scope(scope): #, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def resize_and_rescale_image(image, height, width,
                             do_mean_subtraction=True, scope=None):
    """Prepare one image for training/evaluation.

    Args:
        image: 3-D float Tensor
        height: integer
        width: integer
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor of prepared image.
    """
    if True:
    #with tf.name_scope(values=[image, height, width], name=scope,
                   #    default_name='resize_image'):
        image = tf.expand_dims(image, 0)
        image = tfa.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        if do_mean_subtraction:
            # rescale to [-1,1]
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
    return image


def preprocess_for_train(image,
                         height,
                         width,
                         bbox,
                         max_angle=15.,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=False,do_unprocess=False, **kwargs):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Additionally it would create image_summaries to display the different
  transformations applied to the image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
      bi-cubic resizing, random_hue or random_contrast).
    scope: Optional scope for name_scope.
    add_image_summaries: Enable image summaries.
  Returns:
    3-D float Tensor of distorted image used for training with range [-1, 1].
  """
  if True:
  # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):

    assert image.dtype == tf.float32
    # random rotatation of image between -15 to 15 degrees with 0.75 prob
    angle = random.uniform(-max_angle, max_angle) \
            if random.random() < 0.75 else 0.
    rotated_image = tfa.image.rotate(image, math.radians(angle),
                                            interpolation='BILINEAR')
    # random cropping
    distorted_image, distorted_bbox = distorted_bounding_box_crop(
        rotated_image,
        bbox,
        min_object_covered=0.6,
        area_range=(0.6, 1.0))
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize(x, [height, width]), #todo , method),
        num_cases=num_resize_cases)

    #if add_image_summaries:
    #  tf.summary.image('training_image',
    #                   tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors. There are 1 or 4 ways to do it.
    num_distort_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=num_distort_cases)

    #if add_image_summaries:
    #  tf.summary.image('final_distorted_image',
    #                   tf.expand_dims(distorted_image, 0))


    # preprocessing = 'default'
    # if 'preprocessing' in kwargs:
    #     preprocessing = kwargs['preprocessing']
    # if preprocessing == 'keras_resnet50':
    #     distorted_image = tf.cast(distorted_image * (256.), tf.float32)
    #     distorted_image = tf.keras.applications.resnet50.preprocess_input(distorted_image)
    # else:
    #     distorted_image = tf.subtract(distorted_image, 0.5)
    #     distorted_image = tf.multiply(distorted_image, 2.0)
    if do_unprocess:
          image, _ = unprocess.unprocess(distorted_image, do_mosaic=False)
    return rescale(distorted_image,**kwargs)


def preprocess_for_eval(image,
                        height,
                        width,
                        margin=0,
                        scope=None,
                        add_image_summaries=False,do_unprocess=False, **kwargs):
  """Prepare one image for evaluation.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  if True: #with tf.name_scope(scope, 'eval_image', [image, height, width]):
    assert image.dtype == tf.float32
    #image = resize_and_rescale_image(image, 256, 256,
    #                                 do_mean_subtraction=False) todo: dig why we removed this
    # if False: #steps that we are skipping for now
    image = _aspect_preserving_resize(image, margin+max(height, width))
    image = _central_crop([image], height, width,**kwargs)[0]
    image.set_shape([height, width, 3])
    # else:
    #     image = _aspect_preserving_resize(image, margin+max(height, width))
    #     image = _central_crop([image], height, width)
    #     image = tf.convert_to_tensor(image)
    #     image.set_shape([1,height, width, 3])
    #if add_image_summaries:
    #  tf.summary.image('validation_image', tf.expand_dims(image, 0))
    mirc_crop = None
    if 'mirc_crop' in kwargs:
        mirc_crop = kwargs['mirc_crop']
    if mirc_crop is not None:
        image = _central_crop([image], mirc_crop, mirc_crop)[0]
        image = _aspect_preserving_resize(image, height)

    # def rescale(x):
    #     preprocessing = 'default'
    #     if 'preprocessing' in kwargs:
    #         preprocessing = kwargs['preprocessing']
    #     if preprocessing == 'keras_resnet50':
    #         x = tf.cast(x*(256.), tf.float32)
    #         x = tf.keras.applications.resnet50.preprocess_input(x)
    #     if preprocessing == 'keras_mobilenet_v2':
    #         x = tf.cast(x * (256.), tf.float32)
    #         x = tf.keras.applications.mobilenet.preprocess_input(x)
    #     else:
    #         x = tf.subtract(x, 0.5)
    #         x = tf.multiply(x, 2.0)
    #     return x
    if do_unprocess:
        image, _ = unprocess.unprocess_debug(image, do_mosaic=False,**kwargs)
    return rescale(image, **kwargs)


def preprocess_for_eval_n_steps(image,
                        high_res,
                        low_res,
                        margin=0,
                        # n_steps=0,
                        scope=None,
                        relative_to_initial_offset=False,
                        enforce_zero_initial_offset=False,
                        varying_max_amp=False,
                        add_image_summaries=False, **kwargs):
  """Prepare one image for evaluation.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  if True: #with tf.name_scope(scope, 'eval_image', [image, height, width]):
    assert image.dtype == tf.float32
    #image = resize_and_rescale_image(image, 256, 256,
    #                                 do_mean_subtraction=False)
    n_steps = kwargs.pop('n_steps')
    amp = kwargs['amp']

    centered_offsets = False
    if  'centered_offsets' in kwargs.keys():
        centered_offsets = kwargs['centered_offsets']

    return_position_info = False
    if  'return_position_info' in kwargs.keys():
        return_position_info = kwargs['return_position_info']

    offsets_ = kwargs.pop('offsets')
    if offsets_ is not None:
        if np.ndim(offsets_) == 3:
            offsets = tf.convert_to_tensor(offsets_)[tf.random.uniform(shape=(1,), minval=0, maxval=np.shape(offsets_)[0], dtype=tf.int32)[0]]
        else:
            offsets = offsets_
    else:
        if varying_max_amp:
            max_amp = tf.random.uniform(shape=(n_steps, 2), minval=0, maxval=amp + 1, dtype=tf.int32)
        else:
            max_amp = amp
        offsets = tf.random.uniform(shape=(n_steps, 2), minval=-max_amp, maxval=max_amp + 1, dtype=tf.int32)
        if centered_offsets:
            offsets = offsets - tf.expand_dims(tf.cast(tf.reduce_mean(offsets,axis=0),tf.int32),0)
        elif enforce_zero_initial_offset:
            # enforced by translation of the initial offset to zero rather than by just enforcing zero at the first element.
            # This way we ensure that the overall trajectory statistics is preserved (caveat: modulo clipping that ensures no exceeding margins)
            offsets = offsets - offsets[0]

    offsets = tf.clip_by_value(offsets, clip_value_min=-amp, clip_value_max=amp)

    # print('debug aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa amp',amp)
    image = _aspect_preserving_resize(image, margin+high_res)
    if kwargs['rggb_mode']:
        # image, metadata = unprocess.unprocess(image)
        # shot_noise, read_noise = unprocess.random_noise_levels()
        # low_res_frames = unprocess.add_noise(image, shot_noise, read_noise)
        low_res_frames, high_res_image = _central_crop_with_offsets_rggb(image, high_res, low_res,offsets, n_steps,**kwargs)
    else:
        low_res_frames, high_res_image = _central_crop_with_offsets(image, high_res, low_res,offsets, n_steps,**kwargs)

    offsets_out = offsets
    if relative_to_initial_offset:
            offsets_out = offsets - offsets[0]
    # offsets
    if return_position_info:
        return (rescale(low_res_frames, enforce_def=True,**kwargs), offsets_out), rescale(high_res_image,**kwargs)
    else:
        return rescale(low_res_frames, enforce_def=True,**kwargs), rescale(high_res_image,**kwargs)

def preprocess_image(image,
                     height,
                     width,
                     is_training=False, teacher_mode=False, **kwargs):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image.
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    fast_mode: Optional boolean, if True avoids slower transformations.
    teacher_mode: enforce no multistep, override n_steps

  Returns:
    3-D float Tensor containing an appropriately scaled image
  """
  multistep = -1
  if 'n_steps' in kwargs.keys() and not teacher_mode:
      multistep = kwargs['n_steps']

  rggb_mode = False
  if 'rggb_mode' in kwargs.keys():
      rggb_mode = kwargs['rggb_mode']

  kwargs_upd = kwargs
  if rggb_mode:
      kwargs_upd['preprocessing'] = 'identity'

  if is_training:
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])
    preprocessed_image = preprocess_for_train(image, height, width, bbox, fast_mode=True, **kwargs_upd)
  else:
    preprocessed_image = preprocess_for_eval(image, height, width, **kwargs_upd)

  if rggb_mode:
      preprocessed_image = tf.image.resize(preprocessed_image, [height*2, width*2],method='bilinear')
      preprocessed_image, _ = unprocess.unprocess(preprocessed_image, do_mosaic=True,**kwargs_upd)

      shot_noise, read_noise = unprocess.random_noise_levels()
      preprocessed_image = unprocess.add_noise(preprocessed_image, shot_noise, read_noise)
      preprocessed_image = rescale(preprocessed_image)

  return preprocessed_image
  # old simple version
  # if is_training:
  #   bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
  #                      dtype=tf.float32,
  #                      shape=[1, 1, 4])
  #   return preprocess_for_train(image, height, width, bbox, fast_mode=True)
  # else:
  #   return preprocess_for_eval(image, height, width)


def preprocess_image_drc(image,
                     high_res,
                     low_res,
                     is_training=False, teacher_mode=False, amp=0, **kwargs):
    """Pre-process one image for training or evaluation.

    Args:
    image: 3-D Tensor [height, width, channels] with the image.
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    fast_mode: Optional boolean, if True avoids slower transformations.
    teacher_mode: enforce no multistep, override n_steps

    Returns:
    4-D tensor of low resolution frames and a 3-D float Tensor containing an appropriately scaled image
    """

    low_res_frames, high_res_image = preprocess_for_eval_n_steps(image,
                        high_res,
                        low_res,
                        margin=2*amp+2,
                        amp=amp,
                        **kwargs)
    return low_res_frames, high_res_image

#Example of gaussian blurring
  # psf_sigma = 4
  # filter_shape = (3*psf_sigma,3*psf_sigma)
  # image = tfa.image.gaussian_filter2d(image,sigma=psf_sigma,filter_shape=filter_shape)