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
        # if using a custom model - must specify custom_split_layer


    if parameters['split_after_layer'] is None:
        split_after_layer = default_split_after_layer
    else:
        split_after_layer = parameters['split_after_layer']



    teacher.compile(loss="categorical_crossentropy",
                    metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
                    )

    fe_model, be_model = split_model(teacher, split_after_layer)


    #%%
    #################### Get Layer features as a dataset ##########################
    print('making feature data')
    intermediate_layer_model = fe_model
    batch_size = parameters['batch_size']
    start = 0
    end = batch_size
    train_data = []
    validation_data = []
    upsample_factor = parameters['upsample'] if parameters['upsample'] !=0 else 1
    # train_data = np.zeros([50000,upsample_factor*res,upsample_factor*res,num_feature])
    count = 0

    print('\nLoaded feature data from teacher')

    #%%
    # feature_val_data = train_data[45000:]
    # feature_train_data = train_data[:45000]

    #%%
    ##################### Define Student #########################################
    verbose =parameters['verbose']
    evaluate_prediction_size = 150


    def dummy_metric(y_true, y_pred):
        return 1

    #%%

if parameters['student_version']==3:
    student_fun = student3
elif parameters['student_version'] == 301:
    student_fun = full_size_rggb_fe
else:
    raise ValueError('unknown student version')

print('initializing student')
custom_metrics = []
if parameters['loss'] == 'feature_loss_SKLD':
    with open(parameters['reference_feature_stats'],'rb') as f:
        reference_feature_stats = pickle.load(f)
    if num_features != len(reference_feature_stats['stats']):
        raise ValueError
    loss = keras_loss_for_soft_binning(np.array(reference_feature_stats['stats']),
                                       num_features,
                                       reference_feature_stats['bin_edges'],
                                       tau=0.1,
                                       loss_type='SKLD')

elif parameters['loss'] == 'feature_loss_SKLD_sparse2D':
    with open(parameters['reference_feature_stats'],'rb') as f:
        reference_feature_stats = pickle.load(f)
    if num_features != len(reference_feature_stats['stats']):
        raise ValueError
    loss1D = keras_loss_for_soft_binning(np.array(reference_feature_stats['stats']),
                                       num_features,
                                       reference_feature_stats['bin_edges'],
                                       tau=0.1,
                                       loss_type='SKLD',
                                         custom_name='loss_bins_1D')
    loss2D = keras_loss_for_soft_binning(np.array(reference_feature_stats['stats_2D']),
                                       num_features,
                                       reference_feature_stats['bin_edges_2D'],
                                         couplings=reference_feature_stats['couplings'],
                                       tau=0.1,
                                       loss_type='SKLD',
                                       mode='sparse2D',
                                         custom_name='loss_bins_2D')
    def loss(y1,y2):
        return float(parameters['loss_coeffs'][0])*loss1D(y1,y2) + float(parameters['loss_coeffs'][1])*loss2D(y1,y2)
    custom_metrics = [loss1D, loss2D]
else:
    loss = parameters['loss']

drc_fe_args = dict(sample = parameters['n_samples'],
                   res = res,
                    activation = parameters['student_nl'],
                    dropout = dropout,
                    rnn_dropout = rnn_dropout,
                    num_features = num_features,
                   rnn_layer1 = parameters['rnn_layer1'],
                   rnn_layer2 = parameters['rnn_layer2'],
                   layer_norm = parameters['layer_norm_student'],
                   batch_norm = parameters['batch_norm_student'],
                   conv_rnn_type = parameters['conv_rnn_type'],
                   block_size = parameters['student_block_size'],
                   add_coordinates = parameters['broadcast'],
                   time_pool = parameters['time_pool'],
                   dense_interface=parameters['dense_interface'],
                    loss=loss,
                    upsample=parameters['upsample'],
                    pos_det=parameters['pos_det'],
                    enable_inputB=enable_inputB,
                    expanded_inputB = False,
                      # reference_net=fe_model,
                    rggb_ext_type=parameters['rggb_ext_type'],
                    channels=3 if parameters['rggb_ext_type']==0 else 4,
                    updwn=1 if parameters['rggb_ext_type']==0 else 2,
                    kernel_size=parameters['kernel_size'],
                    custom_metrics=custom_metrics,
                    teacher_only_mode=parameters['teacher_only_at_low_res'],
                    teacher_net_initial_weight = parameters['teacher_net_initial_weight'],
                    teacher_preprsocessing=parameters['preprocessing']
#[debug_metric,debug_metric2,debug_metric3,debug_metric4,debug_metric41,debug_metric42,debug_metric43],
                      #only used for control net 210 (as of Nov.11)
                      )
print(drc_fe_args)

student = student_fun(**drc_fe_args,
                      teacher_net= fe_model if parameters['use_teacher_net_at_low_res'] else None
                      #teacher_net was not passed in the main dictionary of arguments.
                      # Because it is not pickable and probably not savable
                      )
student.summary()

train_accur = []
test_accur = []
# generator parameters:

ctrl_mode = parameters['student_version'] > 100
position_dim = (parameters['max_length'],parameters['res'],parameters['res'],2) if  parameters['broadcast']==1 else (parameters['n_samples'],2)
# movie_dim = (parameters['max_length'], parameters['res'], parameters['res'], 3)  if parameters['student_version'] < 100 else (parameters['res'], parameters['res'], 3)
if parameters['rggb_mode']:
    movie_dim = (parameters['max_length'], parameters['res']//2, parameters['res']//2, 4)
    position_dim = (parameters['max_length'], 2)
else:
    movie_dim = (parameters['max_length'], parameters['res'], parameters['res'], 3)

def args_to_dict(**kwargs):
    return kwargs


if parameters['manual_trajectories']:
    c1 = parameters['trajectory_file'] is not None
    c2 = parameters['trajectories_num'] > 0
    if c1:
        offsets = np.load(parameters['trajectory_file'] )
    if c2:
        offsets = np.random.randint(size=(parameters['trajectories_num'],parameters['n_samples'], 2), low=-parameters['amp'], high=parameters['amp'] + 1, dtype=np.int32)
    if c1 == c2:
        print('debug', c1,c2,parameters['trajectories_num'],parameters['trajectory_file'])
        error('incorrectly defined trajectories')
else:
    offsets = None


generator_params = args_to_dict( n_steps=parameters['n_samples'],
                      feature_net=fe_model, preprocessing=parameters['preprocessing'],rggb_mode=parameters['rggb_mode'],
                       return_position_info=enable_inputB, offsets = offsets, low_res = parameters['res'],
                      unprocess_high_res=parameters['unprocess_high_res'],enable_random_gains=parameters['enable_random_gains'],
                    rggb_teacher=parameters['baseline_rggb_mode'], central_squeeze_and_pad_factor=parameters['central_squeeze_and_pad_factor'],varying_max_amp=parameters['varying_max_amp'])

# load_status = student.load_weights(parameters['pretrained_student_path'])
# decoder =  keras.models.load_model(parameters['pretrained_decoder_path'])
def nets_to_eval(nets_file):
    jj = 0
    with open(nets_file) as f:
        for ii, line in enumerate(f):
            field, value = line.split('=')
            if field.strip() == 'fe_path':
                fe_path = value.strip()
            elif field.strip() == 'decoder_path':
                decoder_path = value.strip()
            elif field.strip() == 'legend':
                legend = value.strip()
            # else: todo, fix this
            #     jj += 1
            if (ii-jj)%3 == 2:
                yield legend,fe_path,decoder_path
# amps = [0,1,3,4,6,8,10,12]
# amps = [0,1,3,4,6,8,10,12, 14, 18 , 22, 28, 36]
amps = [36, 46, 56, 76, 100]
for legend, this_student_path, this_decoder_path in nets_to_eval(parameters['nets_to_eval']):
    load_status = student.load_weights(this_student_path)
    decoder =  keras.models.load_model(this_decoder_path)

    input0 = keras.layers.Input(shape=movie_dim)
    if enable_inputB:
        input1 = keras.layers.Input(shape=position_dim)
        x = student((input0, input1) if parameters['student_version'] < 100 else input0)
        x = decoder(x)
        fro_student_and_decoder = keras.models.Model(inputs=[input0,input1], outputs=x, name='frontend')
    else:
        x = student(input0)
        x = decoder(x)
        fro_student_and_decoder = keras.models.Model(inputs=[input0], outputs=x, name='frontend')
    fro_student_and_decoder.compile(
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
    )
    for this_amp in amps:
        val_generator_classifier = get_dataset(parameters['dataset_dir'], 'validation', batch_size, mode='low_res_with_labels',centered_offsets=False,
                                            enforce_zero_initial_offset=parameters['enforce_zero_initial_offset_cls'], amp=this_amp, **generator_params)
        print(legend, 'evaluated amp', this_amp)
        fro_student_and_decoder.evaluate(val_generator_classifier, steps=1000)

print('\nFinished')

# fe_post_training = fro_student_and_decoder.layers[-2]
# python train.py --student_nl relu --dropout 0.0 --rnn_dropout 0.0 --conv_rnn_type lstm --n_samples 4 --max_length 4 --epochs 10 --int_epochs 0 --teacher_model keras_mobilenet_v2 --student_block_size 3 --time_pool average_pool --student_version 3 --resnet_mode --decoder_optimizer SGD --val_set_mult 1 --res 56 --verbose 2 --broadcast 0 --centered_offsets --amp 2 --rggb_ext_type 0 --kernel_size 3 --no-rggb_mode --num_feature 96 --stu_steps_per_epoch 10000 --preprocessing keras_mobilenet_v2 --skip_student_training --decoder_epochs 0 --skip_final_saves --pretrained_decoder_path ../../toeldad/noname_j36667120_t1677070345_feature/noname_j36667120_t1677070345_saved_models/decoder_trained_model_fix0 --pretrained_student_path ../../toeldad/noname_j36667120_t1677070345_feature/noname_j36667120_t1677070345_feature_net_ckpt