#!/usr/bin/python3
# -*- coding: utf-8 -*-
from sklearn.preprocessing import label_binarize
from log.logger import Logger


class Generator(object):

    def __init__(self,
                 features=None,
                 bath_size=0,
                 labels=None,
                 max_pos=0,
                 classes=0,
                 ):
        self.logger = Logger("Data Loader")
        self._features = features
        self._batch_size = bath_size
        self._labels = labels
        self._max_pos = max_pos
        self._classes = classes

    def generator(self):
        self.logger.info_log("Initialize Generator")
        self._pos = 0
        while True:
            features = self._features[self._pos:self._pos + self._batch_size]
            labels = self._labels[self._pos:self._pos + self._batch_size]
            labels_binary = label_binarize(labels, self._classes)
            self._pos += self._batch_size
            if self._pos >= self._max_pos:
                self._pos = 0
            """
            assert features.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} " \
                "should be {1} at pos {2} " \
                "of max_pos {3}". \
                format(features.shape[0], self._batch_size, self._pos,
                       self._max_pos)
            assert labels_binary.shape[0] == self._batch_size, \
                "in Train Generator features of wrong shape is {0} " \
                "should be {1} at pos {2} " \
                "of max_pos {3}". \
                format(labels.shape[0], self._batch_size,
                       self._pos, self._max_pos)
            """
            yield features, labels_binary
