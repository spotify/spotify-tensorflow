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

import sys
import logging
import multiprocessing as mp

import tensorflow as tf

flags = tf.flags


class Flags(object):
    """spotify_tensorflow flags"""

    @staticmethod
    def register_dataset_flags():
        logging.info("Registering Dataset flags")

        flags.DEFINE_string("train-subdir", "train",
                            "Location of training TFRecords, with the training set dir.")

        flags.DEFINE_string("eval-subdir", "eval",
                            "Location of eval TFRecords, with the training set dir.")

        # Dataset API parameters

        flags.DEFINE_integer("batch-size", 128,
                             "Size of the batch of the dataset iterator. 0 means no batching.")

        flags.DEFINE_integer("shuffle-buffer-size", 512,
                             "Size of the shuffle buffer. 0 means shuffle is turned off.")

        flags.DEFINE_integer("take-count", -1,
                             "Creates a `Dataset` with at most `count` batches from this dataset.")

        flags.DEFINE_integer("parsing-threads", mp.cpu_count(),
                             "Number of threads used for parsing files.")

        flags.DEFINE_integer("interleaving-threads", 2,
                             "Interleaving cycle length. 0 means interleaving is turned off.")

        flags.DEFINE_integer("interleaving-block-length", 32,
                             "Interleaving block length.")

        flags.DEFINE_integer("prefetch-buffer-size", 1024,
                             "Prefetch records. 0 means no pre-fetching.")

    @staticmethod
    def register_core_flags():
        logging.info("Registering core spotify-tensorflow flags")
        flags.DEFINE_string("training-set", None,
                            "Location of the training set")

        flags.DEFINE_string("job-dir", None,
                            "Where to write data")

    @staticmethod
    def register_flags():
        Flags.register_core_flags()
        Flags.register_dataset_flags()

    @staticmethod
    def fail_on_legacy_flags():
        # Flags have been renamed in 0.3.0, old ones would be silently ignored
        _legacy_flags = {
            "train_subdir",
            "eval_subdir",
            "batch_size",
            "shuffle_buffer_size",
            "take_count",
            "parsing_threads",
            "interleaving_threads",
            "interleaving_block_length",
            "prefetch_buffer_size",
            "training_set",
            "job_dir"
        }
        # [RY] Tried to use argparse to parse flags, unsuccessfully :\
        for a in sys.argv[1:]:
            if "=" in a:
                a = a.split("=")[0]
            if a.startswith("--"):
                a = a.strip().lstrip("--")
                if a in _legacy_flags:
                    raise Exception("Shouldn't use legacy flag: " + a)


Flags.fail_on_legacy_flags()
Flags.register_flags()
