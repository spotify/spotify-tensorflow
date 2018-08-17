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

from __future__ import absolute_import, division, print_function

import os

import pandas as pd
from spotify_tensorflow.dataset import Datasets
from examples.examples_utils import get_data_dir


def transform_labels(label_data):
    # Labels are one-hot encoded. Transform them to back to string labels
    return pd.Series.idxmax(label_data)


def main():
    # Enable eager execution for DataFrame endpoint
    import tensorflow as tf
    tf.enable_eager_execution()

    # Set up training data
    train_data_dir = get_data_dir("train")
    train_data = os.path.join(train_data_dir, "part-*")
    schema_path = os.path.join(train_data_dir, "_inferred_schema.pb")

    df_train_data = next(Datasets.dataframe.examples_via_schema(train_data,
                                                                schema_path,
                                                                batch_size=1024))

    # the feature keys are ordered alphabetically for determinism
    label_keys = sorted([l for l in set(df_train_data.columns) if l.startswith("class_name")])
    feature_keys = sorted(set(df_train_data.columns).difference(label_keys))

    label = df_train_data[label_keys].apply(transform_labels, axis=1)
    features = df_train_data[feature_keys]

    # Build model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(multi_class="multinomial", solver="newton-cg")
    model.fit(features, label)

    # Set up eval data
    eval_data_dir = get_data_dir("eval")
    eval_data = os.path.join(eval_data_dir, "part-*")
    df_eval_data = next(Datasets.dataframe.examples_via_schema(eval_data,
                                                               schema_path,
                                                               batch_size=1024))

    eval_label = df_eval_data[label_keys].apply(transform_labels, axis=1)
    eval_features = df_eval_data[feature_keys]

    # Evaluate model
    score = model.score(eval_features, eval_label)
    print("Score is %f" % score)


if __name__ == "__main__":
    main()
