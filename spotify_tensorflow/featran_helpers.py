# -*- coding: utf-8 -*-
#
#  Copyright 2017 Spotify AB.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io

FLAGS = tf.flags.FLAGS


def get_feature_columns(dataset=FLAGS.training_set,
                        label="target",
                        feature_desc_filename="_feature_desc"):
    p = os.path.join(dataset, feature_desc_filename)
    with file_io.FileIO(p, "r") as f:
        return [tf.feature_column.numeric_column(l.strip())
                for l in f.readlines()
                if l.strip() != label]
