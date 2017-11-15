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

import argparse

from spotify_tensorflow.dataset import Datasets
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


class FeatranDatasetExample(object):

    @staticmethod
    def __parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input",
            type=str,
            help="Input directory with Featran output",
            required=True
        )
        parser.add_argument(
            "--show-nbr",
            type=int,
            help="Number of records to show",
            default=2,
            required=True
        )
        arguments, _ = parser.parse_known_args()
        return arguments

    @staticmethod
    def get_input(args):
        with tf.name_scope("input"):
            dataset, c = Datasets.get_featran_example_dataset(args.input, gen_spec=["label"])
            iterator = dataset.make_initializable_iterator()
            (label,), features = iterator.get_next()
            label = tf.reshape(label, [-1, 1])
            features = tf.reshape(features, [-1, c.num_features])
            return iterator, label, features

    @staticmethod
    def eval(iterator, label, features, nbr):
        with tf.name_scope("debug"):
            sess = tf.Session()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            with sess.as_default():
                sess.run(iterator.initializer)
                for i in range(nbr):
                    (_label, _features_) = sess.run([label, features])
                    print("label:", _label)
                    print("features:", _features_)

    def main(self):
        args = FeatranDatasetExample.__parse_args()
        (iterator, label, features) = FeatranDatasetExample.get_input(args)
        FeatranDatasetExample.eval(iterator, label, features, args.show_nbr)


if __name__ == "__main__":
    FeatranDatasetExample().main()
