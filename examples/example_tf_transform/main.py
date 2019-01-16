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
import os
import shutil
import tempfile
import time
from os.path import join as pjoin

import taxi
import tensorflow as tf
import tensorflow_transform as tft
from examples.examples_utils import get_taxi_data_dir
from spotify_tensorflow.tfx.tft import TFTransform


def preprocessing_fn(inputs):
    out = dict()

    for key in taxi.DENSE_FLOAT_FEATURE_KEYS:
        # Preserve this feature as a dense float, setting nan's to the mean.
        out[taxi.transformed_name(key)] = tft.scale_to_z_score(
            taxi.fill_in_missing(inputs[key]))

    for key in taxi.VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        out[taxi.transformed_name(key)] = tft.compute_and_apply_vocabulary(
            taxi.fill_in_missing(inputs[key]), top_k=10, num_oov_buckets=10)

    for key in taxi.BUCKET_FEATURE_KEYS:
        out[taxi.transformed_name(key)] = tft.bucketize(taxi.fill_in_missing(inputs[key]),
                                                        num_buckets=10)

    for key in taxi.CATEGORICAL_FEATURE_KEYS:
        out[taxi.transformed_name(key)] = taxi.fill_in_missing(inputs[key])

    # Was this passenger a big tipper?
    taxi_fare = taxi.fill_in_missing(inputs[taxi.FARE_KEY])
    tips = taxi.fill_in_missing(inputs[taxi.LABEL_KEY])
    out[taxi.transformed_name(taxi.LABEL_KEY)] = tf.where(
        tf.is_nan(taxi_fare),
        tf.cast(tf.zeros_like(taxi_fare), tf.int64),
        # Test if the tip was > 20% of the fare.
        tf.cast(tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64)
    )

    return out


if __name__ == "__main__":
    output_dir_name = "tft_output_%s" % int(time.time())

    taxi_data = get_taxi_data_dir()
    temp = tempfile.mkdtemp()
    os.mkdir(pjoin(temp, "tmp"))
    os.mkdir(pjoin(temp, "output"))
    args = [
        "--training_data=%s" % pjoin(taxi_data, "part-*"),
        "--output_dir=%s" % pjoin(temp, "output"),
        "--schema_file=%s" % pjoin(taxi_data, "chicago_taxi_schema.pbtxt"),
        "--temp_location=%s" % pjoin(temp, "tmp"),
        "--runner=DirectRunner"
    ]

    TFTransform(preprocessing_fn=preprocessing_fn).run(args=args)
    shutil.rmtree(temp)
