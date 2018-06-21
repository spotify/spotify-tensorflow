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

from spotify_tensorflow.dataset import Datasets
import tensorflow as tf
import xgboost as xgb
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
#tf.enable_eager_execution()

def read_data(dirname, label_column='label'):
    from examples_utils import get_data_dir

    raw_data = Datasets.dataframe.read_dataset(get_data_dir(dirname))
    feature_columns = [c for c in raw_data.columns if c != label_column]
    feature_data = raw_data.loc[:, feature_columns]
    label_data = raw_data.loc[:, [label_column]]
    return xgb.DMatrix(label_data, label=label_data)

def main(_):
    param = {'silent':1, 'objective':'binary:logistic', 'booster':'gblinear',
             'alpha': 0.0001, 'lambda': 1}

    training_data = read_data('train')
    eval_data = read_data('eval')

    watchlist = [(eval_data, 'eval'), (training_data, 'train')]
    bst = xgb.train(param, training_data, 1000, watchlist)
    preds = bst.predict(eval_data)
    labels = eval_data.get_label()

    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) /
    float(len(preds))))

if __name__ == "__main__":
    tf.app.run(main=main)
