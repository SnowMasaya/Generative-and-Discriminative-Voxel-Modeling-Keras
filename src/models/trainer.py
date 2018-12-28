#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from log.logger import Logger
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping
from os import path
import os
APP_ROOT = path.dirname(path.abspath("__file__"))
COMMON_PATH = "/./"


class Trainer(object):
    def __init__(self,
                 Model,
                 dataset_name,
                 classes,
                 model_parameter: {},
                 ):
        self.logger = Logger("Trainer")
        self.Model = Model.model
        self._classes = classes
        self._lr_schedule = LearningRateScheduler(LearningRateScheduler)
        self._optimizer = model_parameter['optimizer']
        self.model_parameter = model_parameter
        self.early_stopping = EarlyStopping(patience=10,)
        self.Model_checkpoint = ModelCheckpoint(APP_ROOT + COMMON_PATH +
                                                "/../models/" + dataset_name +
                                                "_epoch_{epoch:02d}_acc_{acc:.2f}.hdf5",
                                                monitor="acc",
                                                verbose=0,
                                                save_best_only=True,
                                                mode="auto")
        model_info = Model.__class__.__name__ + '_'
        print('model_info:', model_info)
        for element, value in model_parameter.items():
            if element == 'class_weight':
                model_info += str(element)
            else:
                model_info += str(element) + '_' + str(value) + '_'
        self.model_info = model_info
        with tf.name_scope("loss"):
            self.Model.compile(loss='categorical_crossentropy',
                               optimizer=self._optimizer,
                               metrics=["accuracy"])
        self.__make_tensorboard()

    def fit_generator(self,
                      generator,
                      steps_per_epoch,
                      fit_generator_parameter: {
                          "epochs": 100,
                          "batch_size": 32,
                          "verbose": 1,
                          "callbacks": None,
                          "validation_data": None,
                          "nb_valid_samples": None,
                          "class_weight": None,
                          "max_q_size": 10,
                          "nb_worker": 1,
                          "initial_epoch": 0,
                      }
                      ):
        callbacks = [self.tensorboard,
                     self.Model_checkpoint,
                     self.early_stopping
                     ]
        self.Model.summary()
        self.Model.fit_generator(
            generator,
            steps_per_epoch,
            fit_generator_parameter['epochs'],
            verbose=fit_generator_parameter['verbose'],
            callbacks=callbacks,
            validation_data=fit_generator_parameter['validation_data'],
            validation_steps=fit_generator_parameter['validation_steps'],
            class_weight=self.model_parameter['class_weight'],
            max_queue_size=fit_generator_parameter['max_q_size'],
            workers=fit_generator_parameter['nb_worker'],
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=fit_generator_parameter['initial_epoch']
            )
        self.__save_model(self.model_info)

    def __save_model(self, name):
        self.Model.save(name + ".h5")

    def evaluate(self, input_data, labels, batch_size=32, verbose=1,
                 sample_weight=None):

        self._score = self.Model.evaluate(input_data,
                                          labels,
                                          batch_size=batch_size,
                                          verbose=verbose,
                                          sample_weight=sample_weight)
        self.logger.info_log("Test score {0}".format(self._score))
        return self._score

    def __make_tensorboard(self):
        """
        For make tensorboard to visualize result
        :param batch_size: setting batch_size
        """
        directory_name = self.model_info
        if os.path.isdir(APP_ROOT + COMMON_PATH + "/log/") is False:
            os.mkdir(APP_ROOT + COMMON_PATH + "/log/" + directory_name)
        if os.path.isdir(APP_ROOT + COMMON_PATH + "/log/" + directory_name) is False:
            os.mkdir(APP_ROOT + COMMON_PATH + "/log/" + directory_name)
        self.tensorboard = \
            TensorBoard(log_dir=APP_ROOT + COMMON_PATH + "/log/log_data/" +
                                directory_name,
                        write_graph=True,
                        )
