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

import logging

import tensorflow as tf

flags = tf.flags


class Flags(object):
    """spotify_tensorflow flags"""

    @staticmethod
    def register_dataset_flags():
        logging.info("Registering Dataset flags")
        flags.DEFINE_integer("batch_size", 128,
                             "Size of the batch of the dataset iterator.")

        flags.DEFINE_integer("buffer_size", 512,
                             "Size of the buffer of the dataset iterator.")

        flags.DEFINE_integer("take_count", -1,
                             "Creates a `Dataset` with at most `count` batches from this dataset.")

        flags.DEFINE_string("train_subdir", "train",
                            "Location of training TFRecords, with the training set dir.")

        flags.DEFINE_string("eval_subdir", "eval",
                            "Location of eval TFRecords, with the training set dir.")

    @staticmethod
    def register_core_flags():
        logging.info("Registering core spotify-tensorflow flags")
        flags.DEFINE_string("training_set", None,
                            "Location of the training set")

        flags.DEFINE_string("job-dir", None,
                            "Where to write data")

    @staticmethod
    def register_flags():
        Flags.register_core_flags()
        Flags.register_dataset_flags()


Flags.register_flags()
