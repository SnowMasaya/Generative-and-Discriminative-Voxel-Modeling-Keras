#!/usr/bin/python3
# -*- coding: utf-8 -*-
from os import path
import os
import pickle
from GPyOpt.methods import BayesianOptimization
from keras.optimizers import Adam
from data.generator import Generator
from models.voxception_resnet_deep import ModelVoxCeptionResDeep4Net
from models.trainer import Trainer
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
APP_ROOT = path.dirname(path.abspath('__file__'))


space = [
    {'name': 'learning_rate', 'type': 'continuous',
     'domain': (0.0008, 0.002), 'dimensionality': 1},
    {'name': 'batch_size', 'type': 'discrete',
     'domain': (50, 100, 150,), 'dimensionality': 1},
    {'name': 'drop_out', 'type': 'discrete',
     'domain': (0.1, 0.2, 0.3, 0.4), 'dimensionality': 1},
]

def data_load(feature_file_name, label_file_name):
    with open(feature_file_name, 'rb') as f:
        feature_data = pickle.load(f)

    feature_data = np.swapaxes(feature_data, 1, 2)
    feature_data = np.swapaxes(feature_data, 2, 3)
    feature_data = np.swapaxes(feature_data, 3, 4)

    with open(label_file_name, 'rb') as f:
        label_data = pickle.load(f)

    return feature_data, label_data

def hyper_parameter_tune(x):

    params = {}
    params['learning_rate'] = float(x[:, 0])
    adam = Adam(lr=params['learning_rate'],
                beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    params['optimizer'] = adam
    params['optimizer_str'] = 'adam'

    params['batch_size'] = int(x[:, 1])
    params['drop_out'] = float(x[:, 2])
    params['train_val_split'] = "stratify"

    print(params)

    train_feature_name = '../data/processed/features_train.pkl'
    train_label_file_name = '../data/processed/labels_train.pkl'
    valid_feature_name = '../data/processed/features_valid.pkl'
    valid_label_file_name = '../data/processed/labels_valid.pkl'
    test_feature_name = '../data/processed/features_test.pkl'
    test_label_file_name = '../data/processed/labels_test.pkl'

    train_feature, train_label = data_load(train_feature_name,
                                          train_label_file_name,
                                          )

    valid_feature, valid_label  = data_load(valid_feature_name,
                                            valid_label_file_name,
                                            )

    test_feature, test_label = data_load(test_feature_name,
                                         test_label_file_name,
                                         )

    concat_feature = np.append(train_feature, valid_feature, axis=0)
    concat_label = np.append(train_label, valid_label, axis=0)

    concat_feature = np.append(concat_feature, test_feature, axis=0)
    concat_label = np.append(concat_label, test_label, axis=0)

    print('concat_feature:', concat_feature.shape)
    print('concat_label:', concat_label.shape)

    X_train_tmp, X_test, y_train_tmp, y_test = train_test_split(
        concat_feature,
        concat_label,
        test_size=0.2,
        random_state=42,
        stratify=concat_label,
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_tmp,
        y_train_tmp,
        test_size=0.2,
        random_state=42,
        stratify=y_train_tmp,
        )

    print('X_valid:', X_valid.shape)
    print('y_valid:', y_valid.shape)

    print('train_feature:', train_feature.shape)

    class_number = np.unique(concat_label)

    print('class_number:', class_number)

    train_generator = Generator(
        features=X_train,
        bath_size=params['batch_size'],
        labels=y_train,
        max_pos=X_train.shape[0],
        classes=class_number
    )

    valid_generator = Generator(
        features=X_valid,
        bath_size=params['batch_size'],
        labels=y_valid,
        max_pos=X_valid.shape[0],
        classes=class_number,
    )

    model = ModelVoxCeptionResDeep4Net(len(class_number))
    print(model.__class__.__name__)
    model_trainer = Trainer(
                            Model=model,
                            dataset_name="ModelVoxCeptionResDeep4Net",
                            classes=len(class_number),
                            model_parameter=params,
    )

    fit_generator_parameter = {
        "epochs": 100,
        "batch_size": params['batch_size'],
        "verbose": 1,
        "callbacks": None,
        "validation_data": valid_generator.generator(),
        "validation_steps": X_valid.shape[0] // params['batch_size'],
        "class_weight": None,
        "max_q_size": 10,
        "nb_worker": 4,
        "initial_epoch": 0,
    }

    model_trainer.fit_generator(
        generator=train_generator.generator(),
        steps_per_epoch=train_feature.shape[0] // params['batch_size'],
        fit_generator_parameter=fit_generator_parameter,
    )

    labels_binary_test_random = label_binarize(y_test, class_number)
    evaluate_score = model_trainer.evaluate(
        X_test, labels_binary_test_random,
        batch_size=params['batch_size'],
        verbose=1,)

    return evaluate_score[1]


if __name__ == '__main__':
    myBopt = BayesianOptimization(f=hyper_parameter_tune,
                                  domain=space)

    myBopt.run_optimization(max_iter=100)
