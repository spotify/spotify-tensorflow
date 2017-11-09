# -*- coding: utf-8 -*-
#
# Copyright 2017 Spotify AB.
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

import os

import tensorflow as tf

from .dataset_lib import Datasets

FLAGS = tf.flags.FLAGS


def mk_dataset_training(training_set):
    with tf.name_scope('traininginput'):
        train_dataset, _ = Datasets.get_featran_example_dataset(os.path.join(training_set,
                                                                             FLAGS.train_subdir))
        return train_dataset


def mk_dataset_eval(training_set):
    with tf.name_scope('evalinput'):
        eval_dataset, _ = Datasets.get_featran_example_dataset(os.path.join(training_set,
                                                                            FLAGS.eval_subdir))
        return eval_dataset
