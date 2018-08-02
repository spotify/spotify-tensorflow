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

import pandas as pd
from spotify_tensorflow.dataset import Datasets
from examples.examples_utils import get_data_dir, iris_features


def transform_labels(label_data):
    # Labels are one-hot encoded. Transform them to back to string labels
    return pd.Series.idxmax(label_data)


def main():
    # Set up training data
    train_data_dir = get_data_dir("train")
    df_train_data = Datasets.dataframe.read_dataset(train_data_dir,
                                                    feature_mapping_fn=iris_features)
    feature_context = Datasets.get_context(train_data_dir)
    (feature_names, label_names) = feature_context.multispec_feature_groups

    label = df_train_data.loc[:, label_names].apply(transform_labels, axis=1)
    features = df_train_data.loc[:, feature_names]

    # Set up eval data
    eval_data_dir = get_data_dir("eval")
    df_eval_data = Datasets.dataframe.read_dataset(eval_data_dir,
                                                   feature_mapping_fn=iris_features)
    eval_label = df_eval_data.loc[:, label_names].apply(transform_labels, axis=1)
    eval_features = df_eval_data.loc[:, feature_names]

    # Build model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(multi_class="multinomial", solver="newton-cg")
    model.fit(features, label)

    # Evaluate model
    score = model.score(eval_features, eval_label)
    print("Score is %f" % score)


if __name__ == "__main__":
    main()
