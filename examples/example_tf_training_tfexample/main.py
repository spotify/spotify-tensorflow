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

from spotify_tensorflow import Datasets, Trainer
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    from examples_utils import get_data_dir
    import tempfile

    config = Trainer.get_default_run_config(tempfile.mkdtemp())

    feature_context = Datasets.get_context(get_data_dir("train"))
    (feature_names, label_names) = feature_context.multispec_feature_groups
    features = [tf.feature_column.numeric_column(x) for x in feature_names]

    def split_features_label_fn(spec):
        label = spec.pop(label_names[0])
        return spec, label

    classifier = tf.estimator.LinearClassifier(features, config=config)
    Trainer.run(estimator=classifier,
                training_data_dir=get_data_dir("train"),
                eval_data_dir=get_data_dir("eval"),
                split_features_label_fn=split_features_label_fn,
                run_config=config)


if __name__ == "__main__":
    tf.app.run(main=main)
