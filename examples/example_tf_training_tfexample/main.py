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
from spotify_tensorflow.trainer import Trainer
import tensorflow as tf
import multiprocessing as mp
import os

tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.contrib.data import make_batched_features_dataset
import tensorflow as tf
from tensorflow.python import debug as tf_debug




def main(_):
    from examples_utils import get_data_dir
    import tempfile

    config = Trainer.get_default_run_config("/tmp/job1")

    feature_context = Datasets.get_context(get_data_dir("train"))
    (feature_names, label_names) = feature_context.multispec_feature_groups
    features = [tf.feature_column.numeric_column('f3')]


    def split_features_label_fn(spec):
        label = spec.pop(label_names[0])
        spec.pop('f1')
        spec.pop('f2_ODD')
        spec.pop('f2_EVEN')
        return spec, label

    def train_input_fn():
        # A lot happens here, we:
        # * read tf.Examples stored in tf.Records
        # * parse Examples using features spec from feature_context.features
        # * we parallelize reading and parsing
        # * we don't shuffle, there is distributed shuffle in Label job
        # * we batch and repeat whole training dataset epoch times
        d = make_batched_features_dataset(os.path.join(get_data_dir("train"), "part-*"),
                                          batch_size=32,
                                          num_epochs=1,
                                          shuffle=False,
                                          reader_num_threads=2,
                                          parser_num_threads=2,
                                          features=feature_context.features)
        return d.map(split_features_label_fn, num_parallel_calls=mp.cpu_count())

    def eval_input_fn():
        d = make_batched_features_dataset(os.path.join(get_data_dir("eval"), "part-*"),
                                          batch_size=32,
                                          num_epochs=1,
                                          shuffle=False,
                                          reader_num_threads=2,
                                          parser_num_threads=2,
                                          features=feature_context.features)
        return d.map(split_features_label_fn, num_parallel_calls=mp.cpu_count())

    classifier = tf.estimator.LinearClassifier(features, config=config)
    hook = tf_debug.TensorBoardDebugHook("Fallons-MacBook-Pro-3.local:7001")
    classifier.train(input_fn=train_input_fn, hooks=[hook])
    eval_metrics = classifier.evaluate(input_fn=eval_input_fn)

if __name__ == "__main__":
    tf.app.run(main=main)
