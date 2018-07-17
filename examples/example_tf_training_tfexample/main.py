# -*- coding: utf-8 -*-
#
# Copyright 2018 Spotify AB.
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

import tensorflow as tf
from spotify_tensorflow.dataset import Datasets

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def train(_):
    from examples_utils import get_data_dir
    import tempfile

    config = tf.estimator.RunConfig(tempfile.mkdtemp())

    training_data_dir = get_data_dir("train")
    feature_context = Datasets.get_context(training_data_dir)

    (feature_names, label_names) = feature_context.multispec_feature_groups
    features = [tf.feature_column.numeric_column(x) for x in feature_names]

    def split_features_label_fn(spec):
        # Canned TF's LinearClassifier requires label to be a single integer, Featran gives us
        # one hot encoding for class, thus we need to convert one hot encoding to single integer
        labels = tf.concat([[spec.pop(l)] for l in label_names], axis=0)
        label = tf.argmax(labels, axis=0)
        # Get the rest of the features out of the spec
        return spec, label

    def get_in_fn(dir):
        def in_fn():
            train_input_it, _ = Datasets.mk_iter(dir)
            return split_features_label_fn(train_input_it.get_next())
        return in_fn

    classifier = tf.estimator.LinearClassifier(feature_columns=features,
                                               n_classes=3,
                                               config=config)

    classifier.train(get_in_fn(get_data_dir("train")))\
        .evaluate(get_in_fn(get_data_dir("eval")))


def main():
    tf.app.run(main=train)


if __name__ == "__main__":
    main()
