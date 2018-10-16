# -*- coding: utf-8 -*-
#
# Copyright 2018 Spotify AB.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import json
from collections import OrderedDict
from os.path import join as pjoin
from typing import Callable, List, Dict, Any, Union, Iterator  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from tensorflow.python.lib.io import file_io


class Featran(object):

    @classmethod
    def settings(cls, settings_dir, settings_filename=None):
        # type: (str, str) -> List[Dict[str, Any]]
        """
        Read a Featran settings file and return a list of settings

        :param settings_dir: Path to the directory containing the settings file
        :param settings_filename: Filename of the Featran Settings JSON file
        :return: A List of Featran Settings
        """
        f = cls.__get_featran_settings_file(settings_dir, settings_filename)
        with file_io.FileIO(f, "r") as fio:
            settings = json.load(fio)
        return settings

    @classmethod
    def names(cls, settings_path, feature_splitter_fn=None):
        # type: (str, Callable[[Any], str]) -> Union[List[str], Dict[str, List[str]]]
        """
        Returns a list of Featran feature names.  Optionally the list of names
        can be split into a dictionary keyed by the feature_splitter_fn

        :param settings_path: Path to the Featran Settings JSON Directory
        :param feature_splitter_fn: Function to split feature name into a keyed dictionary
        :return: A List or dictionary of Featran Feature names
        """
        settings = cls.settings(settings_path)
        if feature_splitter_fn:
            return cls.__split_names(settings, feature_splitter_fn)
        else:
            return cls.__all_names(settings)

    @classmethod
    def reorder_numpy_dataset(cls,
                              dataset,  # type: Iterator[Dict[str, np.ndarray]]
                              settings_path  # type: str
                              ):
        # type: (...) -> Iterator[OrderedDict[str, np.ndarray]]
        """
        Reorders a numpy dictionary so that feature keys are in the same order as those in a Featran
        settings file.

        :param dataset: A dataset created via Datasets.dict
        :param settings_path: Path to the Featran Settings JSON Directory
        :return: An iterator over an OrderedDict mapping feature names to Numpy arrays
        """
        feature_names = Featran.names(settings_path)
        for batch in dataset:
            yield OrderedDict((name, batch[name]) for name in feature_names)

    @classmethod
    def reorder_dataframe_dataset(cls,
                                  dataset,  # type: Iterator[pd.DataFrame]
                                  settings_path  # type: str
                                  ):
        # type: (...) -> Iterator[pd.DataFrame]
        """
        Reorders a pandas DataFrame so that feature columns are in the same order as those in a
        Featran settings file.

        :param dataset: A dataset created via Datasets.dataframe
        :param settings_path: Path to the Featran Settings JSON Directory
        :return: An iterator over new DataFrame batches with ordered columns
        """
        feature_names = Featran.names(settings_path)
        for batch in dataset:
            yield batch[feature_names]

    @staticmethod
    def __get_featran_settings_file(dir_path, settings_filename=None):
        # type: (str, str) -> str
        filename = settings_filename if settings_filename else "part-00000-of-00001.txt"
        filepath = pjoin(dir_path, filename)
        assert file_io.file_exists(filepath), "settings file `%s` does not exist" % filepath
        return filepath

    @staticmethod
    def __split_names(settings, feature_splitter_fn):
        # type: (List[Dict[str, Any]], Callable[[Any], str]) -> Dict[str, List[str]]
        from collections import defaultdict
        feature_names = defaultdict(list)  # type: Dict[str, List[str]]
        for setting in settings:
            key = feature_splitter_fn(setting["name"])
            for name in setting["featureNames"]:
                feature_names[key].append(name)
        return feature_names

    @staticmethod
    def __all_names(settings):
        # type: (List[Dict[str, Any]]) -> List[str]
        feature_names = []
        for setting in settings:
            for name in setting["featureNames"]:
                feature_names.append(name)
        return feature_names
