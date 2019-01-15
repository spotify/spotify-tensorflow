# -*- coding: utf-8 -*-
#
# Copyright 2017-2019 Spotify AB.
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

from __future__ import absolute_import

from collections import defaultdict
import logging
import os

import numpy as np
import six
import tensorflow as tf
import xgboost as xgb
from spotify_tensorflow.dataset import Datasets
from examples.examples_utils import get_data_dir


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("rounds", 50, "Number of Rounds")
flags.DEFINE_string("local_dir", "/tmp/features", "GCS Train Path")


def transform_dataset(dataset, features_keys):
    data = defaultdict(list)

    for key, values in six.iteritems(dataset):
        if key in features_keys:
            data["features"].append(values)
        else:
            data["labels"].append(values)

    return list(zip(*data["features"])), np.argmax(list(six.moves.zip(*data["labels"])), axis=1)


def train(_):
    # Set up training data
    train_data_dir = get_data_dir("train")
    train_data = os.path.join(train_data_dir, "part-*")
    schema_path = os.path.join(train_data_dir, "_inferred_schema.pb")

    training_dataset = next(Datasets.dict.examples_via_schema(train_data,
                                                              schema_path,
                                                              batch_size=1024))

    # the feature keys are ordered alphabetically for determinism
    label_keys = sorted([l for l in training_dataset.keys() if l.startswith("class_name")])
    features_keys = sorted(set(training_dataset.keys()).difference(label_keys))

    (feature_train_data, labels_train_data) = transform_dataset(training_dataset, features_keys)

    eval_data_dir = get_data_dir("eval")
    eval_data = os.path.join(eval_data_dir, "part-*")
    eval_dataset = next(Datasets.dict.examples_via_schema(eval_data,
                                                          schema_path,
                                                          batch_size=1024))
    (feature_eval_data, labels_eval_data) = transform_dataset(eval_dataset, features_keys)

    params = {
        "objective": "multi:softmax",
        "verbose": False,
        "num_class": len(label_keys),
        "max_depth": 6,
        "nthread": 4,
        "silent": 1
    }

    xg_train = xgb.DMatrix(feature_train_data,
                           label=labels_train_data)

    xg_eval = xgb.DMatrix(feature_eval_data,
                          label=labels_eval_data)

    watchlist = [(xg_eval, "eval"), (xg_train, "train")]
    xg_model = xgb.train(params, xg_train, FLAGS.rounds, watchlist)

    preds = xg_model.predict(xg_eval)
    labels = xg_eval.get_label()
    error = (sum(1 for i in range(len(preds)) if preds[i] != labels[i]) /
             float(len(preds)))
    print("Score: %f" % (1.0 - error))


def main():
    tf.enable_eager_execution()
    tf.app.run(main=train)


if __name__ == "__main__":
    main()
