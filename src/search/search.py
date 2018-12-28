#!/usr/bin/python3
# -*- coding: utf-8 -*-
from annoy import AnnoyIndex
import numpy as np
from keras.models import load_model


class Search(object):

    def __init__(self,
                 featture: int=40,
                 tree: int=10,
                 regist_annoy: str="feature.ann",
                 model_name: str="",):
        self.annoy_index = AnnoyIndex(featture)
        self.tree = tree
        self.regist_annoy = regist_annoy
        self.model_name = model_name

    def search(self,
               feature: np.ndarray=np.ndarray,
               nearest: int=10):
        self.annoy_index.load(self.regist_annoy)
        result = self.annoy_index.get_nns_by_vector(feature, nearest,
                                           search_k=-1,
                                           include_distances=False)
        return result

    def regist(self, feature_list: np.ndarray):
        for index, feature in enumerate(feature_list):
            print("Total {} index {}".format(feature_list.shape[0], index))
            self.annoy_index.add_item(index, feature)
        self.annoy_index.build(self.tree)
        self.annoy_index.save(self.regist_annoy)

    def extract_feature(self, features: np.ndarray):
        model = load_model(self.model_name)

        # remove last layer
        model.layers.pop()
        model.outputs = [model.layers[-2].output]
        model.layers[-1].outbound_nodes = []

        extract_feature_set = np.ndarray([])

        for index, each_test_feature in enumerate(features):
            print("extract index: ", index)
            each_test_feature = np.reshape(each_test_feature,
                                           (1,
                                           each_test_feature.shape[1],
                                           each_test_feature.shape[2],
                                           each_test_feature.shape[3],
                                           1
                                           ))
            predict_data = model.predict(each_test_feature)
            if index == 0:
                extract_feature_set = predict_data
            else:
                extract_feature_set = np.vstack((extract_feature_set,
                                                 predict_data))

        return extract_feature_set
