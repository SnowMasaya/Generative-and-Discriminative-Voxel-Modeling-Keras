#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from types import MappingProxyType


class ModelVoxNet(object):

    def __init__(self, number_class,
                 parameter_dict={
                     'input_shape': (32, 32, 32, 1),
                     'filter_size': 32,
                     'conv_kernel_size_1': 5,
                     'conv_kernel_size_2': 3,
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

        with tf.name_scope('Model'):
            with tf.name_scope('Inputs'):
                inputs = Input(default_parameter_dict['input_shape'])

            # convolution 1

            count_dict['Conv3D'] = 1

            with tf.name_scope('Conv3D_' + str(count_dict['Conv3D'])):
                x = Conv3D(filters=default_parameter_dict['filter_size'],
                           kernel_size=default_parameter_dict[
                               'conv_kernel_size_1'],
                           padding='valid',
                           strides=default_parameter_dict['strides_1'],
                           data_format='channels_last',
                           )(inputs)

            # Activation Leaky ReLu

            count_dict['Activation_relu'] = 1

            with tf.name_scope(
                    'Activation_relu_' + str(count_dict['Activation_relu'])):
                x = Activation('relu')(x)

            # dropout 1

            count_dict['Dropout'] = 1

            with tf.name_scope('Dropout_' + str(count_dict['Dropout'])):
                x = Dropout(
                    rate=default_parameter_dict['drop_out_rate_list'][0])(x)

            # convolution 2

            count_dict['Conv3D'] += 1

            with tf.name_scope('Conv3D_' + str(count_dict['Conv3D'])):
                x = Conv3D(filters=default_parameter_dict['filter_size'],
                           kernel_size=default_parameter_dict[
                               'conv_kernel_size_2'],
                           padding='valid',
                           strides=default_parameter_dict['strides_2'],
                           data_format='channels_last',
                           )(x)

            # Activation Leaky ReLu
            count_dict['Activation_relu'] += 1

            with tf.name_scope(
                    'Activation_relu_' + str(count_dict['Activation_relu'])):
                x = Activation('relu')(x)

            # max pool 1
            with tf.name_scope('MaxPooling3D'):
                x = MaxPooling3D(pool_size=default_parameter_dict['pool_size'],
                                 strides=None,
                                 padding='valid',
                                 data_format='channels_last',
                                 )(x)

            # dropout 2
            count_dict['Dropout'] += 1

            with tf.name_scope('Dropout_' + str(count_dict['Dropout'])):
                x = Dropout(
                    rate=default_parameter_dict['drop_out_rate_list'][2])(x)

            # dense 1 (fully connected layer)

            with tf.name_scope('Flatten'):
                x = Flatten()(x)

            count_dict['Dense'] = 1

            with tf.name_scope('Dense'):
                x = Dense(units=default_parameter_dict['units'],
                          activation='linear', )(x)

            # dropout 3
            count_dict['Dropout'] += 1

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
