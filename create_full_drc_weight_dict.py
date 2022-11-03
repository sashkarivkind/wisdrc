#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
an auxillary function that loads a full DRC model from the front end and back end parts and saves it as a single pickle file
usage example:
python create_full_drc_weight_dict.py
--pretrained_decoder_path ../drc_saves/saved_models/noname_j_t1667498515/final_weights/decoder_final_model/ <folder name, for the decoder model>
--pretrained_student_path ../drc_saves/saved_models/noname_j_t1667498515/final_weights/fe_final_weights/fe_final_weights <file name of the decoder weights>
--student_args ../drc_saves/saved_models/noname_j_t1667498515/student_parameters/student_args.pkl <input pickle file with student parameters, if not specified, then you can specify the parameters via standard command line options>
--out_weight_pickle_file <where to output the dictionary>
"""

import os 
import sys
import gc
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import time
import pickle
from models.student_models import  student3

from tensorflow.keras.applications.resnet50 import ResNet50
from utils.split_keras_model import split_model
from parse_commandline import parse_commandline

print(os.getcwd() + '/')

config = parse_commandline()

print('config  ',config)

parameters = config
TESTMODE = parameters['testmode']

num_feature = parameters['num_feature']
n_samples = parameters['n_samples']
res = parameters['res']
student_block_size = parameters['student_block_size']
print(parameters)
enable_inputB = parameters['broadcast'] != 0

#%%
############################### Get Trained Teacher ##########################3

path = os.getcwd() + '/'
if True:
    if parameters['teacher_net'] is not None:
        teacher = keras.models.load_model(parameters['teacher_net'])

    else:
        if parameters['teacher_model']=='keras_resnet50':
            teacher = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                  include_top=True,
                                                                  weights='imagenet')
            split_after_layer = 'pool1_pool'

        elif parameters['teacher_model']=='keras_mobilenet_v2':
            teacher = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                                                  include_top=True,
                                                                  weights='imagenet')
            split_after_layer = 'block_1_depthwise_BN'
        else:
            error

    teacher.compile(loss="categorical_crossentropy",
                    metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
                    )

    fe_model, be_model = split_model(teacher, split_after_layer)


    #%%
    #################### Get Layer features as a dataset ##########################
    print('making feature data')
    intermediate_layer_model = fe_model
    decoder = be_model

    upsample_factor = parameters['upsample'] if parameters['upsample'] !=0 else 1

    ##################### Define Student #########################################
if parameters['student_version']==3:
    student_fun = student3
elif parameters['student_version'] == 4:
    student_fun = student4
elif parameters['student_version'] == 5:
    student_fun = student5
elif parameters['student_version'] == 103:
    student_fun = student_ctrl103
elif parameters['student_version'] == 210:
    student_fun = student_ctrl210
else:
    error

print('initializing student')
if parameters['student_args'] is not None:
    with open(parameters['student_args'], 'rb') as f:
        drc_fe_args = pickle.load(f)
else:
    drc_fe_args = dict(sample = parameters['n_samples'],
                       res = res,
                        activation = parameters['student_nl'],
                        dropout = dropout,
                        rnn_dropout = rnn_dropout,
                        num_feature = num_feature,
                       rnn_layer1 = parameters['rnn_layer1'],
                       rnn_layer2 = parameters['rnn_layer2'],
                       layer_norm = parameters['layer_norm_student'],
                       batch_norm = parameters['batch_norm_student'],
                       conv_rnn_type = parameters['conv_rnn_type'],
                       block_size = parameters['student_block_size'],
                       add_coordinates = parameters['broadcast'],
                       time_pool = parameters['time_pool'],
                       dense_interface=parameters['dense_interface'],
                        loss=parameters['loss'],
                          upsample=parameters['upsample'],
                          pos_det=parameters['pos_det'],
                        enable_inputB=enable_inputB,
                        expanded_inputB = False,
                          # reference_net=fe_model,
                        rggb_ext_type=parameters['rggb_ext_type'],
                          channels=3 if parameters['rggb_ext_type']==0 else 4,
                        updwn=1 if parameters['rggb_ext_type']==0 else 2,
                        kernel_size=parameters['kernel_size'],
                        custom_metrics=[]
                          )
    print(drc_fe_args)
    with open('last_run_student_args.pkl','wb') as f:
        pickle.dump(drc_fe_args,f)


student = student_fun(**drc_fe_args)
student.summary()

def weights_to_np_dict(net):
    return {layer.name: layer.get_weights() for layer in net.layers}


load_status1 = student.load_weights(parameters['pretrained_student_path'])
decoder2 = keras.models.load_model(parameters['pretrained_decoder_path'])

drc_weights = {'fe': weights_to_np_dict(student), 'be': weights_to_np_dict(decoder2)}

with open(parameters['out_weight_pickle_file'],'wb') as f:
    pickle.dump(drc_weights,f)

for layer in student.layers:
    layer.set_weights(drc_weights['fe'][layer.name])

for layer in decoder2.layers:
    layer.set_weights(drc_weights['be'][layer.name])

print('mazal_tov!!!')

