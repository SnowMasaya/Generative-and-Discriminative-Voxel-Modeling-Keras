#!/usr/bin/python3
# -*- coding: utf-8 -*-
from unittest import TestCase
from os import path
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from data.preprocess import Preprocess
APP_ROOT = path.dirname(path.abspath('__file__'))


class TestPreprocess(TestCase):
    def test_preprceoss(self):
        data_path = APP_ROOT + "/test/test_data/"
        self.preprocess = Preprocess()
        result_feature, result_label = self.preprocess.preprceoss(
            data_path=data_path)

        assert result_feature.shape == (9, 1, 32, 32, 32)
        assert result_label.shape == (9,)
