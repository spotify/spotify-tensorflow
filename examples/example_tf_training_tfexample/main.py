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

import os
from collections import OrderedDict

import tensorflow as tf
from spotify_tensorflow.dataset import Datasets
from examples.examples_utils import get_data_dir


tf.logging.set_verbosity(tf.logging.INFO)


def train(_):
    import tempfile

    config = tf.estimator.RunConfig(tempfile.mkdtemp())

    train_data_dir = get_data_dir("train")
    schema_path = os.path.join(train_data_dir, "_inferred_schema.pb")

    feature_spec, _ = Datasets.parse_schema(schema_path)
    # we use OrderedDict and sorted keys for features for determinism
    all_features = OrderedDict([(name, tf.feature_column.numeric_column(name, default_value=.0))
                                for name in sorted(feature_spec.keys())])
    feature_columns = all_features.copy()
    label_keys = sorted([l for l in set(feature_columns.keys()) if l.startswith("class_name")])
    for l in label_keys:
        feature_columns.pop(l)

    def split_features_label_fn(spec):
        # Canned TF's LinearClassifier requires label to be a single integer, Featran gives us
        # one hot encoding for class, thus we need to convert one hot encoding to single integer
        tf_major_ver = int(tf.__version__.split(".")[0])
        if(tf_major_ver == 0):
            labels = tf.concat([[spec.pop(l)] for l in label_keys], concat_dim=0)
        else:
            labels = tf.concat([[spec.pop(l)] for l in label_keys], axis=0)
        label = tf.argmax(labels, axis=0)
        # Get the rest of the features out of the spec
        return spec, label

    def get_in_fn(data):
        raw_feature_spec = tf.feature_column.make_parse_example_spec(all_features.values())

        def in_fn():
            dataset = Datasets.examples_via_feature_spec(data, raw_feature_spec)
            return dataset.map(split_features_label_fn)
        return in_fn

    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns.values(),
                                               n_classes=3,
                                               config=config)

    train_data = os.path.join(train_data_dir, "part-*")
    eval_data = os.path.join(get_data_dir("eval"), "part-*")
    classifier.train(get_in_fn(train_data)).evaluate(get_in_fn(eval_data))


def main():
    tf.app.run(main=train)


if __name__ == "__main__":
    main()
