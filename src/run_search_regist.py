#!/usr/bin/python3
# -*- coding: utf-8 -*-
from os import path
import pickle
from search.search import Search
APP_ROOT = path.dirname(path.abspath('__file__'))


def data_load(feature_file_name, label_file_name):
    with open(feature_file_name, 'rb') as f:
        feature_data = pickle.load(f)

    with open(label_file_name, 'rb') as f:
        label_data = pickle.load(f)

    return feature_data, label_data


def regist():
    test_feature_name = '../data/processed/features_test.pkl'
    test_label_file_name = '../data/processed/labels_test.pkl'

    test_feature, test_label = data_load(test_feature_name,
                                         test_label_file_name,
                                         )

    search_instance = Search(
        featture=512,
        model_name=APP_ROOT + "/../models/model_net_epoch_19_acc_0.96.hdf5",
        regist_annoy=APP_ROOT + "/../models/feature_model_net_epoch_19_acc_0.96_all_test_data.ann",
    )

    extract_features = search_instance.extract_feature(test_feature)

    search_instance.regist(extract_features)


if __name__ == '__main__':
    regist()
