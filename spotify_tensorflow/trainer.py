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

from .experiment import mk_experiment_fn

FLAGS = tf.flags.FLAGS


class Trainer(object):
    """Entry point to train/evaluate estimators."""

    @staticmethod
    def __split_features_label_fn(parsed_features):
        target = parsed_features.pop("target")
        return parsed_features, target

    @staticmethod
    def __get_default_training_data_dir():
        from os.path import join as pjoin
        return pjoin(FLAGS.training_set, FLAGS.train_subdir)

    @staticmethod
    def __get_default_eval_data_dir():
        from os.path import join as pjoin
        return pjoin(FLAGS.training_set, FLAGS.eval_subdir)

    @staticmethod
    def __get_default_run_config():
        return tf.contrib.learn.RunConfig(model_dir=FLAGS.job_dir)

    @staticmethod
    def run(estimator,
            training_data_dir=None,
            eval_data_dir=None,
            split_features_label_fn=None,
            run_config=None):
        """Make and run an experiment based on given estimator.

        Args:
            estimator: Your estimator to train on. See official TensorFlow documentation on how to
                define your own estimator.
            training_data_dir: Directory containing training data.
                Default value is based on `Flags`.
            eval_data_dir: Directory containing training data. Default value is based on `Flags`.
            split_features_label_fn: Function used split features into examples and labels.
            run_config: `RunConfig` for the `Estimator`. Default value is based on `Flags`.
        """

        training_data_dir = training_data_dir or Trainer.__get_default_training_data_dir()
        eval_data_dir = eval_data_dir or Trainer.__get_default_eval_data_dir()
        run_config = run_config or Trainer.__get_default_run_config()

        logging.info("Training data directory: `%s`", training_data_dir)
        logging.info("Evaluation data directory: `%s`", eval_data_dir)

        tf.contrib.learn.learn_runner.run(experiment_fn=mk_experiment_fn(estimator,
                                                                         training_data_dir,
                                                                         eval_data_dir,
                                                                         split_features_label_fn),
                                          run_config=run_config)
