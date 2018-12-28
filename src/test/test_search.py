#!/usr/bin/python3
# -*- coding: utf-8 -*-
from unittest import TestCase
import pickle
from os import path
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from search.search import Search
APP_ROOT = path.dirname(path.abspath('__file__'))


class TestSearch(TestCase):
    def test_extract_feature(self):
        with open(APP_ROOT + "/../data/processed/test_code_feature.pkl",
                  "rb") as f:
            test_feature = pickle.load(f)

        search_instance = Search(model_name=APP_ROOT + "/../models/model_net_epoch_19_acc_0.96.hdf5")

        extract_features = search_instance.extract_feature(test_feature)

        assert extract_features.shape == (100, 512)

    def test_regist(self):
        with open(APP_ROOT + "/../data/processed/test_code_feature.pkl",
                  "rb") as f:
            test_feature = pickle.load(f)

        search_instance = Search(
            featture=512,
            model_name=APP_ROOT + "/../models/model_net_epoch_19_acc_0.96.hdf5",
            regist_annoy=APP_ROOT + "/../models/feature_model_net_epoch_19_acc_0.96.ann",
        )

        extract_features = search_instance.extract_feature(test_feature)

        search_instance.regist(extract_features)

    def test_search(self):
        with open(APP_ROOT + "/../data/processed/test_code_feature.pkl",
                  "rb") as f:
            test_feature = pickle.load(f)

        search_instance = Search(
            featture=512,
            model_name=APP_ROOT + "/../models/model_net_epoch_19_acc_0.96.hdf5",
            regist_annoy=APP_ROOT + "/../models/feature_model_net_epoch_19_acc_0.96.ann",
        )

        extract_features = search_instance.extract_feature(test_feature)

        search_result = search_instance.search(extract_features[0])

        assert search_result == [0, 15, 58, 96, 2, 48, 40, 6, 50, 42]
