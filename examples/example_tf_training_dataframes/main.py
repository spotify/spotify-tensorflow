#!/usr/bin/env python
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

from spotify_tensorflow import Datasets


def main():
    from examples_utils import get_data_dir
    train_data_dir = get_data_dir()

    df_train_data = Datasets.dataframe.read_dataset(train_data_dir)

    # lets divide dict into label and features
    label = df_train_data.pop("label")
    features = df_train_data

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(features, label)

    eval_data_dir = get_data_dir("eval")
    df_eval_data = Datasets.dataframe.read_dataset(eval_data_dir)
    eval_label = df_eval_data.pop("label")
    eval_features = df_eval_data

    score = model.score(eval_features, eval_label)
    print("Score is %f" % score)


if __name__ == "__main__":
    main()
