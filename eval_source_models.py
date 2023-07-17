#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import sys
import gc

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import pickle
from models.student_models import  student3,full_size_rggb_fe

from utils.split_keras_model import split_model
from utils.imagenet_dataset import get_dataset
from utils.saving_utils import mkdir_if_needed
from utils.logging import Logger
from parse_commandline import parse_commandline
from utils.feature_stats import keras_loss_for_soft_binning
print(os.getcwd() + '/')

config = parse_commandline()

# print('config  ',config)

parameters = config
TESTMODE = parameters['testmode']


lsbjob = os.getenv('LSB_JOBID')
save_path = parameters['local_save_path'] if lsbjob is None else parameters['lsb_save_path']

lsbjob = '' if lsbjob is None else lsbjob

num_features = parameters['num_features']
trajectory_index = parameters['trajectory_index']
n_samples = parameters['n_samples']
res = parameters['res']
trajectories_num = parameters['trajectories_num']
run_index = parameters['run_index']
dropout = parameters['dropout']
rnn_dropout = parameters['rnn_dropout']
this_run_name = parameters['run_name_prefix'] + '_j' + lsbjob + '_t' + str(int(time.time()))
lsf_synch_path = parameters['run_name_prefix'] + '_j' + lsbjob + '_lsf_pointers.txt'

parameters['this_run_name'] = this_run_name
epochs = parameters['epochs']
int_epochs = parameters['int_epochs']
student_block_size = parameters['student_block_size']

save_model_path = os.path.join(save_path, 'saved_models', this_run_name)
student_parameters_path = os.path.join(save_model_path, 'student_parameters')
results_path = os.path.join(save_model_path, 'results_path')
student_checkpoint_path = os.path.join(save_model_path, 'student_checkpoint')
fine_tuning_checkpoint_path = os.path.join(save_model_path, 'fine_tuning_checkpoint')
final_weights_path = os.path.join(save_model_path, 'final_weights')
mkdir_if_needed(save_model_path)
mkdir_if_needed(student_parameters_path)
mkdir_if_needed(student_checkpoint_path)
mkdir_if_needed(fine_tuning_checkpoint_path)
mkdir_if_needed(results_path)
student_checkpoint_filename = os.path.join(student_checkpoint_path, 'student_checkpoint')
fine_tuning_checkpoint_filename = os.path.join(fine_tuning_checkpoint_path, 'fine_tuning_checkpoint')

sys.stdout = Logger(os.path.join(results_path, 'log.log'))

print(parameters)
# scale pixels

if parameters['gpu_id'] > -1:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[parameters['gpu_id']], 'GPU')



enable_inputB = parameters['broadcast'] != 0

#%%
############################### Get Trained Teacher ##########################3

path = os.getcwd() + '/'
if True:
    if parameters['teacher_model']=='keras_resnet50':
        teacher = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                              include_top=True,
                                                              weights='imagenet')
        default_split_after_layer = 'pool1_pool'

    elif parameters['teacher_model']=='keras_mobilenet_v2':
        teacher = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                                              include_top=True,
                                                              weights='imagenet')
        default_split_after_layer = 'block_1_depthwise_BN'

    elif parameters['teacher_model']=='VGG16':
        teacher = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                                              include_top=True,
                                                              weights='imagenet')
        default_split_after_layer = 'block2_pool'

    elif parameters['teacher_model']=='VGG19':
        teacher = tf.keras.applications.VGG19(input_shape=(224, 224, 3),
                                                              include_top=True,
                                                              weights='imagenet')
        default_split_after_layer = 'block2_pool'

    else:
        teacher = keras.models.load_model(parameters['teacher_model'])

    teacher.summary()

    teacher.compile(loss="categorical_crossentropy",
                    metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
                    )



    #%%
    #################### Get Layer features as a dataset ##########################

    batch_size = parameters['batch_size']

    verbose =parameters['verbose']
    evaluate_prediction_size = 150



print('preparing generators')

val_generator_baseline = get_dataset(parameters['dataset_dir'], 'validation', batch_size, image_h = parameters['image_h'],image_w = parameters['image_w'],
                                     preprocessing=parameters['preprocessing'],rggb_mode=parameters['baseline_rggb_mode'],
                                     central_squeeze_and_pad_factor=parameters['central_squeeze_and_pad_factor'])



gc.collect()

print('Evaluating teacher network')
teacher.evaluate(val_generator_baseline, steps=10000 // batch_size , verbose = verbose)


'''
evaluation results:
alexnet with alexnet preprocese:
312/312 - 13s - loss: 2.1773 - categorical_accuracy: 0.5112 - top_k_categorical_accuracy: 0.7508
alexnet with resnet preproccwss:
312/312 - 12s - loss: 2.0275 - categorical_accuracy: 0.5356 - top_k_categorical_accuracy: 0.7749
resnet50 with resnet50 perprocessing:
312/312 - 21s - loss: 1.1277 - categorical_accuracy: 0.7368 - top_k_categorical_accuracy: 0.9121

'''