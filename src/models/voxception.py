#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, concatenate
from keras.layers.core import Activation, Dense, Dropout, Flatten
from types import MappingProxyType
from keras import backend


def conv3d_bn(x,
              parameter_dict={
                  "filters": 32,
                  "kernel": 3,
                  "padding": 'valid',
                  "strides": (1, 1, 1),
                  "name": None
              },
              count_dict={}):
    default_parameter_dict = parameter_dict
    default_parameter_dict = MappingProxyType(default_parameter_dict)
    default_parameter_dict = default_parameter_dict
    if default_parameter_dict['name'] is not None:
        bn_name = default_parameter_dict['name'] + '_bn'
        conv_name = default_parameter_dict['name'] + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    count_dict['Conv3D'] += 1
    with tf.name_scope('Conv3D_' + str(count_dict['Conv3D'])):
        x = Conv3D(
            default_parameter_dict["filters"],
            kernel_size=default_parameter_dict['kernel'],
            strides=default_parameter_dict['strides'],
            padding=default_parameter_dict['padding'],
            use_bias=False,
            name=conv_name,
            data_format=backend.image_data_format(),
        )(x)

    count_dict['BatchNormalization'] += 1

    with tf.name_scope(
            'BatchNormalization_' + str(count_dict['BatchNormalization'])):
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    count_dict['Activation_relu'] += 1

    with tf.name_scope(
            'Activation_relu_' + str(count_dict['Activation_relu'])):
        x = Activation('relu', name=default_parameter_dict['name'])(x)
    return x


class ModelVoxCeptionNet(object):

    def __init__(self, number_class,
                 parameter_dict={
                     'input_shape': (32, 32, 32, 1),
                     'filter_size': 32,
                     'conv_kernel_size_1': 5,
                     'conv_padding': 'valid',
                     'bn_conv_kernel_size_1': 3,
                     'bn_conv_kernel_size_2': 1,
                     'bn_conv_padding': 'same',
                     'strides_1': (2, 2, 2),
                     'strides_2': (1, 1, 1),
                     'drop_out_rate_list': [0.3, 0.4, 0.5],
                     'pool_size': (2, 2, 2),
                     'units': 128
                 }
                 ):

        # init model

        default_parameter_dict = parameter_dict
        default_parameter_dict = MappingProxyType(default_parameter_dict)
        self.default_parameter_dict = default_parameter_dict

        count_dict = {}

        data_format = backend.image_data_format()

        if backend.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        with tf.name_scope('Model'):
            with tf.name_scope('Inputs'):
                inputs = Input(default_parameter_dict['input_shape'])

            # convolution 1

            count_dict['Conv3D'] = 1

            with tf.name_scope('Conv3D_' + str(count_dict['Conv3D'])):
                x = Conv3D(filters=default_parameter_dict['filter_size'],
                           kernel_size=default_parameter_dict[
                               'conv_kernel_size_1'],
                           padding=default_parameter_dict['conv_padding'],
                           strides=default_parameter_dict['strides_1'],
                           data_format=data_format,
                           )(inputs)

            count_dict['Activation_relu'] = 0
            count_dict['BatchNormalization'] = 0
            count_dict['BnConv3D'] = 1

            with tf.name_scope('BnConv3D_' + str(count_dict['BnConv3D'])):
                vx_3x3 = conv3d_bn(
                    x,
                    parameter_dict={
                        "filters": int(default_parameter_dict['filter_size']),
                        "kernel": default_parameter_dict[
                            'bn_conv_kernel_size_1'],
                        "padding": default_parameter_dict['bn_conv_padding'],
                        "strides": default_parameter_dict['strides_2'],
                        "name": 'BnConv3D_' + str(count_dict['BnConv3D'])},
                    count_dict=count_dict,
                )

            count_dict['BnConv3D'] += 1

            with tf.name_scope('BnConv3D_' + str(count_dict['BnConv3D'])):
                vx_1x1 = conv3d_bn(
                    x,
                    parameter_dict={
                        "filters": int(default_parameter_dict['filter_size']),
                        "kernel": default_parameter_dict[
                            'bn_conv_kernel_size_2'],
                        "padding": default_parameter_dict['bn_conv_padding'],
                        "strides": default_parameter_dict['strides_2'],
                        "name": 'BnConv3D_' + str(count_dict['BnConv3D'])},
                    count_dict=count_dict,
                )

            count_dict['Concat'] = 1
            with tf.name_scope('Concat_' + str(count_dict['Concat'])):
                x = concatenate([vx_3x3, vx_1x1], axis=-1,
                                name='mixed' + str(count_dict['Concat']))

            # max pool 1
            with tf.name_scope('MaxPooling3D'):
                x = MaxPooling3D(pool_size=default_parameter_dict['pool_size'],
                                 strides=None,
                                 padding='valid',
                                 data_format='channels_last',
                                 )(x)

            # dense 1 (fully connected layer)

            with tf.name_scope('Flatten'):
                x = Flatten()(x)

            count_dict['Dense'] = 1

            with tf.name_scope('Dense'):
                x = Dense(units=default_parameter_dict['units'],
                          activation='linear', )(x)

            # dropout 3
            count_dict['Dropout'] = 1

            with tf.name_scope('Dropout_' + str(count_dict['Dropout'])):
                x = Dropout(
                    rate=default_parameter_dict['drop_out_rate_list'][2])(x)

            # dense 2 (fully connected layer)

            count_dict['Dense'] += 1

            with tf.name_scope('Dense'):
                x = Dense(units=number_class,
                          activation='linear',
                          )(x)

            # Activation Softmax
            with tf.name_scope('Activation_softmax'):
                outputs = Activation("softmax")(x)

            self.model = Model(inputs=[inputs], outputs=[outputs])
            self.model.summary()


