#!/usr/bin/python3
# -*- coding: utf-8 -*-

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class OFFMatrix(object):
    """Representation of OFF dataset to be used instead of a Numpy array.
    # Example
    ```python
        x_data = OFFMatrix('input/file.off', 'data')
        model.predict(x_data)
    ```
    Optionally, a normalizer function (or lambda) can be given. This will
    be called on every slice of data retrieved.
    # Arguments
        datapath: string, path to a OFF file
        dataset: string, name of the OFF dataset in the file specified
            in datapath
        normalizer: function to be called on data when retrieved
    # Returns
        An array-like OFF dataset.
    """

    def __init__(self, data_path: str, normalizer: object=None):
        FILE_FORMAT = "OFF"
        self.read_data = []
        try:
            with open(data_path, "r") as f:
                self.read_data = f.read().strip().split()
        except IOError as io:
            print(str(io))
        self.data = {}
        if self.read_data[0] != FILE_FORMAT:
            self.data["file_format"] = FILE_FORMAT
            self.data["num_vertices"] = \
                int(self.read_data.pop(0).replace(FILE_FORMAT, ""))
        else:
            self.data["file_format"] = self.read_data.pop(0)
            self.data["num_vertices"] = int(self.read_data.pop(0))
        self.data["num_faces"] = int(self.read_data.pop(0))
        self.data["num_edges"] = int(self.read_data.pop(0))
        self.normalizer = normalizer
        VERTICES_DIM = 3
        FACES_DIM = 4
        if self.data["num_vertices"] != 0:
            self.data["vertices"] = self.__get_data_process(
                                    self.data["num_vertices"],
                                    VERTICES_DIM)
        if self.data["num_faces"] != 0:
            self.data["faces"] = self.__get_data_process(
                                 self.data["num_faces"],
                                 FACES_DIM)

    def __get_data_process(self, data_number: int, points: int):
        """
        Get data process
        :param data_number(int): set the getting data number
        :param points(int): set the points such as the verticle or faces
        :return:
        """
        tmp_list = []
        data_array = []
        for i in range(data_number):
            [tmp_list.append(float(self.read_data.pop(0))) for j in range(points)]  # noqa
            data_array.append(tmp_list)
            tmp_list = []
        if self.normalizer is not None:
            return self.normalizer(np.array(data_array, dtype=float))
        else:
            return np.array(data_array)

    def __len__(self):
        return len(self.read_data)

    def __getitem__(self, key: int):
        if isinstance(key, str):
            return self.data[key]
        else:
            raise IndexError
