#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains following models:
student3 - the baseline model, enriched version from the iclr2022 paper
pos_det101 - position detector model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

def student3(sample = 10, res = 8, activation = 'tanh', dropout = 0.0, rnn_dropout = 0.0, upsample = 0,
             num_feature = 1, layer_norm = False ,batch_norm = False, n_layers=3, conv_rnn_type='lstm',block_size = 1,
             add_coordinates = False, time_pool = False, coordinate_mode=1, attention_net_size=64, attention_net_depth=1,
             rnn_layer1=32,
             rnn_layer2=64,
             kernel_size=3,
             dense_interface=False,
            loss="mean_squared_error",
            pos_det=None,
            enable_inputB = True,
             channels=3,
             updwn=2,
             rggb_ext_type=1,
            expanded_inputB=True,
             custom_metrics=[],
             **kwargs):
    #TO DO add option for different block sizes in every convcnn
    #TO DO add skip connections in the block
    #coordinate_mode 1 - boardcast,
    #coordinate_mode 2 - add via attention block
    inputB_ready_flag = False
    if time_pool == '0':
        time_pool = 0
    inputA = keras.layers.Input(shape=(sample, res//updwn,res//updwn,channels))
    if add_coordinates and coordinate_mode==1:
        if expanded_inputB:
            inputB = keras.layers.Input(shape=(sample,res,res,2))
        else:
            inputB = keras.layers.Input(shape=(sample, 2))
            inputB_ = tf.expand_dims(inputB,2)
            inputB_ = tf.expand_dims(inputB_,2)
            inputB_  = tf.tile(inputB_, [1,1, res//updwn,res//updwn,1])
            inputB_ready_flag = True

    else:
        inputB = keras.layers.Input(shape=(sample,2))
    if conv_rnn_type == 'lstm':
        Our_RNN_cell = keras.layers.ConvLSTM2D
    elif  conv_rnn_type == 'gru':
        Our_RNN_cell = ConvGRU2D
    else:
        error("not supported type of conv rnn cell")

    #if a position detector is specified then coordinates are being inferred via this detector
    if pos_det is None:
        if not inputB_ready_flag:
            inputB_ = inputB
    else:
        pos_det_inst = keras.models.load_model(pos_det)
        pos_det_inst.trainable = False
        inputB_ = pos_det_inst(inputA)
        if coordinate_mode == 1:
            # inputB_ = tf.broadcast_to(inputB_,(res,res))#.transpose([0,1,3,4,2])
            # inputB_ = tf.transpose(
            #     tf.tile(inputB_,[1,res,res]),
            #     [0,1,3,4,2])
            inputB_ = tf.expand_dims(inputB_,2)
            inputB_ = tf.expand_dims(inputB_,2)
            inputB_  = tf.tile(inputB_, [1,1, res//updwn,res//updwn,1])
        else:
            error



    #Broadcast the coordinates to a [res,res,2] matrix and concat to x
    if add_coordinates:
        if coordinate_mode==1:
            x = keras.layers.Concatenate()([inputA,inputB_])
        elif coordinate_mode==2:
            x = inputA
            a = keras.layers.GRU(attention_net_size,input_shape=(sample, None),return_sequences=True)(inputB)
            for ii in range(attention_net_depth-1):
                a = keras.layers.GRU(attention_net_size, input_shape=(sample, None), return_sequences=True)(a)
    else:
        x = inputA

    if upsample != 0:
        x = keras.layers.TimeDistributed(keras.layers.UpSampling2D(size=(upsample, upsample)))(x)

    if rggb_ext_type ==1:
        x = keras.layers.TimeDistributed(keras.layers.UpSampling2D(size=(updwn, updwn)))(x)

    print(x.shape)
    for ind in range(block_size):
        x = Our_RNN_cell(rnn_layer1,(kernel_size,kernel_size), padding = 'same', return_sequences=True,
                                dropout = dropout,recurrent_dropout=rnn_dropout,
                            name = 'convLSTM1{}'.format(ind))(x)
    for ind in range(block_size):
        x = Our_RNN_cell(rnn_layer2,(kernel_size,kernel_size), padding = 'same', return_sequences=True,
                            name = 'convLSTM2{}'.format(ind),
                            dropout = dropout,recurrent_dropout=rnn_dropout,)(x)
        if add_coordinates and coordinate_mode==2:
            a_ = keras.layers.TimeDistributed(keras.layers.Dense(64,activation="tanh"))(a)
            a_ = keras.layers.Reshape((sample, 1, 1, -1))(a_)
            x = x * a_
    for ind in range(block_size):
        if ind == block_size - 1:
            if time_pool:
                return_seq = True
            else:
                return_seq = False
        else:
            return_seq = True
        x = Our_RNN_cell(num_feature,(kernel_size,kernel_size), padding = 'same', return_sequences=return_seq,
                            name = 'convLSTM3{}'.format(ind), activation=activation,
                            dropout = dropout,recurrent_dropout=rnn_dropout,)(x)
        if dense_interface:
            if return_seq:
                x = keras.layers.TimeDistributed(keras.layers.Conv2D(num_feature, (3, 3), padding='same',
                                        name='anti_sparse'))(x)
            else:
                x = keras.layers.Conv2D(num_feature, (3, 3), padding='same',
                                        name='anti_sparse')(x)
    print(return_seq)
    if time_pool:
        print(time_pool)
        if time_pool == 'max_pool':
            x = tf.keras.layers.MaxPooling3D(pool_size=(sample, 1, 1))(x)
        elif time_pool == 'average_pool':
            x = tf.keras.layers.AveragePooling3D(pool_size=(sample, 1, 1))(x)
        x = tf.squeeze(x,1)

    if rggb_ext_type == 2:
        x = keras.layers.UpSampling2D(size=(updwn, updwn))(x)
    if rggb_ext_type == 3:
        x = keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")(x)
    if layer_norm:
        x = keras.layers.LayerNormalization(axis=3)(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)

    print(x.shape)
    if enable_inputB:
        model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'student_3')
    else:
        model = keras.models.Model(inputs=[inputA],outputs=x, name = 'student_3')

    opt=tf.keras.optimizers.Adam(lr=1e-3)

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr
    lr_metric = get_lr_metric(opt)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["mean_squared_error", "mean_absolute_error", "cosine_similarity",lr_metric] + custom_metrics,
    )
    return model


def pos_det101(sample = 10, res = 8, activation = 'tanh', dropout = 0.0, rnn_dropout = 0.0, n_layers=3, conv_rnn_type='gru',
            n_units=32, d_filter=3,channels=3,
            loss="mean_squared_error",
               metrics=[], top_lin_layer=False,
             **kwargs):

    inputA = keras.layers.Input(shape=(sample, res,res,channels))

    if conv_rnn_type == 'lstm':
        Our_RNN_cell = keras.layers.ConvLSTM2D
    elif  conv_rnn_type == 'gru':
        Our_RNN_cell = ConvGRU2D
    else:
        error("not supported type of conv rnn cell")

    #Broadcast the coordinates to a [res,res,2] matrix and concat to x
    x = inputA
    for ind in range(n_layers):
        x = Our_RNN_cell(n_units,(d_filter,d_filter), padding = 'same', return_sequences=True,
                                dropout = dropout,recurrent_dropout=rnn_dropout,
                            name = 'convLSTM1{}'.format(ind))(x)
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(x)

    x = keras.layers.GRU(n_units,input_shape=(sample, None),return_sequences=True)(x)

    if top_lin_layer:
        x = keras.layers.TimeDistributed(keras.layers.Dense(2))(x)
    else:
        x = keras.layers.GRU(2, input_shape=(sample, None), return_sequences=True)(x)

    model = keras.models.Model(inputs=inputA,outputs=x, name = 'pos_det')
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["mean_squared_error", "mean_absolute_error", "cosine_similarity"]+metrics,
    )
    return model
