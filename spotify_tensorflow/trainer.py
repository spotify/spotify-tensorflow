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

import tensorflow as tf

from .experiment import mk_experiment_fn
from .run_config import ConfigHelper

FLAGS = tf.flags.FLAGS


def split_features_label_fn(parsed_features):
    target = parsed_features.pop('target')
    return (parsed_features, target)


def run(estimator,
        training_set=FLAGS.training_set,
        split_features_label_fn=split_features_label_fn):
    """
    :return:
    """
    tf.contrib.learn.learn_runner.run(experiment_fn=mk_experiment_fn(estimator,
                                                    training_set,
                                                    split_features_label_fn),
                     run_config=ConfigHelper.run_config())
