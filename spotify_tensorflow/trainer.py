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

from .dataset import Datasets

__all__ = ["Trainer"]

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
    def __get_default_experiment_fn(estimator,
                                    training_data_dir,
                                    eval_data_dir,
                                    feature_mapping_fn,
                                    split_features_label_fn):
        def in_fn():
            train_input_it, _ = Datasets.mk_iter(training_data_dir,
                                                 "evaluation-input",
                                                 feature_mapping_fn)
            return split_features_label_fn(train_input_it.get_next())

        def eval_fn():
            eval_input_it, _ = Datasets.mk_iter(eval_data_dir,
                                                "training-input",
                                                feature_mapping_fn)
            return split_features_label_fn(eval_input_it.get_next())

        def do_make_experiment(run_config, params):
            return tf.contrib.learn.Experiment(
                estimator=estimator,
                train_input_fn=in_fn,
                eval_input_fn=eval_fn)

        return do_make_experiment

    @staticmethod
    def get_default_run_config(job_dir=FLAGS.job_dir):
        """Returns a default `RunConfig` for `Estimator`."""
        # this weird try/except is a static variable pattern in python
        # https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function/16214510#16214510
        try:
            return Trainer.get_default_run_config.default_config
        except AttributeError:
            assert job_dir is not None, "Please pass a non None job_dir"
            Trainer.get_default_run_config.default_config = tf.contrib.learn.RunConfig(
                model_dir=job_dir)
            return Trainer.get_default_run_config.default_config

    @staticmethod
    def run(estimator,
            training_data_dir=None,
            eval_data_dir=None,
            feature_mapping_fn=None,
            split_features_label_fn=None,
            run_config=None,
            experiment_fn=None):
        """Make and run an experiment based on given estimator.

        Args:
            estimator: Your estimator to train on. See official TensorFlow documentation on how to
                define your own estimator.
            training_data_dir: Directory containing training data.
                Default value is based on `Flags`.
            eval_data_dir: Directory containing training data. Default value is based on `Flags`.
            feature_mapping_fn: A function which maps feature spec line to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), tf.int64, default_value=0).
            split_features_label_fn: Function used split features into examples and labels.
            run_config: `RunConfig` for the `Estimator`. Default value is based on `Flags`.
            experiment_fn: Function which returns an `Experiment`. Default value is based on
                `Flags` and is implementation specific.
        """

        training_data_dir = training_data_dir or Trainer.__get_default_training_data_dir()
        eval_data_dir = eval_data_dir or Trainer.__get_default_eval_data_dir()
        run_config = run_config or Trainer.__get_default_run_config()
        experiment_fn = experiment_fn or Trainer.__get_default_experiment_fn(estimator,
                                                                             training_data_dir,
                                                                             eval_data_dir,
                                                                             feature_mapping_fn,
                                                                             split_features_label_fn
                                                                             )

        logging.info("Training data directory: `%s`", training_data_dir)
        logging.info("Evaluation data directory: `%s`", eval_data_dir)

        tf.contrib.learn.learn_runner.run(experiment_fn=experiment_fn,
                                          run_config=run_config)
