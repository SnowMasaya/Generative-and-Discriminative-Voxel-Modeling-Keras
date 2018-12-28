#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import sys
from pathlib import Path
import re
from data.off_matrix import OFFMatrix


class Preprocess(object):
    def __init__(self):
        self.TINY_NUMBER = sys.float_info.epsilon

    def preprceoss(self, data_path: str):

        p = Path(data_path)

        classes = list(p.iterdir())

        off_file_list = list(p.glob('**/*.off'))

        tmp_list = []
        label = []
        for off_file in off_file_list:
            off_file = str(off_file)
            label.append(re.sub("^.*\/|\_[0-9]+|.off", "", off_file))
            off_instance = OFFMatrix(data_path=off_file)
            feature = np.array(self.__voxilize(off_instance.data["vertices"]))
            tmp_list.append(feature)
        return np.array(tmp_list), np.array(label)

    def __voxilize(self, np_pc, rot=None, TEST_CHECK=False):
        """
        this function converts a tango tablet matrix into a voxnet
        voxel volume
        Args:
            np_pc: numpy ndarray with density grid data from load_pc
            rot: ability to rotate picture rot times and take
                 rot recognitions
        Returns:
            voxilized version of density grid that is congruent
            with voxnet size
        """

        max_dist = 0.0
        for it in range(0, 3):
            # find min max & distance in current direction
            min = np.amin(np_pc[:, it])
            max = np.amax(np_pc[:, it])
            dist = max - min

            # find maximum distance
            if dist > max_dist:
                max_dist = dist

            # set middle to 0,0,0
            np_pc[:, it] = np_pc[:, it] - dist / 2 - min

            # covered cells
            CoveredCells = 29

            # find voxel edge size
            vox_sz = dist / (CoveredCells - 1)

            # if 0 divid
            if vox_sz == 0:
                vox_sz = self.TINY_NUMBER

            # render pc to size 30x30x30 from middle
            np_pc[:, it] = np_pc[:, it] / vox_sz

        for it in range(0, 3):
            np_pc[:, it] = np_pc[:, it] + (CoveredCells - 1) / 2

        # round to integer array
        np_pc = np.rint(np_pc).astype(np.uint32)

        # fill voxel arrays
        vox = np.zeros([30, 30, 30])
        for (pc_x, pc_y, pc_z) in np_pc:
            if TEST_CHECK:
                vox[pc_x, pc_y, pc_z] = 1.0
            elif random.randint(0, 100) < 80:
                vox[pc_x, pc_y, pc_z] = 1.0

        np_vox = np.zeros([1, 32, 32, 32])
        np_vox[0, 1:-1, 1:-1, 1:-1] = vox

        return np_vox