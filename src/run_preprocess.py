#!/usr/bin/python3
# -*- coding: utf-8 -*-
from os import path
import pickle
from data.preprocess import Preprocess
from sklearn.model_selection import train_test_split
APP_ROOT = path.dirname(path.abspath('__file__'))


def preprocess():
    # You have to set the model net data
    data_path = APP_ROOT + "/../data/raw/ModelNet40/"
    save_path = APP_ROOT + "/../data/process/"

    preprocess_instance = Preprocess()
    result_feature, result_label = preprocess_instance.preprceoss(
        data_path=data_path)

    X_train_tmp, X_test, y_train_tmp, y_test = train_test_split(
        result_feature,
        result_label,
        test_size=0.2,
        random_state=42,
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_tmp,
        y_train_tmp,
        test_size=0.2,
        random_state=42,
        )

    with open(save_path + 'features_train.pkl') as f:
        pickle.dump(X_train, f,)
    with open(save_path + 'labels_train.pkl') as f:
        pickle.dump(y_train, f,)

    with open(save_path + 'features_valid.pkl') as f:
        pickle.dump(X_valid, f,)
    with open(save_path + 'labels_valid.pkl') as f:
        pickle.dump(y_valid, f,)

    with open(save_path + 'features_test.pkl') as f:
        pickle.dump(X_test, f,)
    with open(save_path + 'labels_test.pkl') as f:
        pickle.dump(y_test, f,)


if __name__ == '__main__':
    preprocess()
