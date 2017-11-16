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

import tensorflow as tf

from .dataset import Datasets

FLAGS = tf.flags.FLAGS


def mk_experiment_fn(estimator,
                     training_data_dir,
                     eval_data_dir,
                     split_features_label):
    """
    :return: tf.contrib.learn.Experiment https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment  # noqa: E501
    """

    def in_fn():
        train_input_dataset = Datasets.mk_dataset_training(training_data_dir)
        train_input_it = mk_iterator(train_input_dataset)
        return split_features_label(train_input_it.get_next())

    def eval_fn():
        eval_input_dataset = Datasets.mk_dataset_eval(eval_data_dir)
        eval_input_it = mk_iterator(eval_input_dataset)
        return split_features_label(eval_input_it.get_next())

    def do_make_experiment(run_config, params):
        return tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=in_fn,
            eval_input_fn=eval_fn)

    return do_make_experiment


def mk_iterator(dataset,
                batch_size=FLAGS.batch_size,
                buffer_size=FLAGS.buffer_size,
                take_count=FLAGS.take_count):
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(take_count)
    iterator = dataset.make_one_shot_iterator()
    return iterator
