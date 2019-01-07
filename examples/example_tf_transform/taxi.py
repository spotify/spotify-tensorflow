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
import tensorflow as tf


def transformed_name(key):
    return key + "_xf"


def fill_in_missing(x):
    default_value = "" if x.dtype == tf.string else 0

    dense_tensor = tf.sparse_to_dense(x.indices,
                                      [x.dense_shape[0], 1],
                                      x.values,
                                      default_value)
    return tf.squeeze(dense_tensor, axis=1)


DENSE_FLOAT_FEATURE_KEYS = ["trip_miles", "fare", "trip_seconds"]

VOCAB_FEATURE_KEYS = ["payment_type", "company"]

BUCKET_FEATURE_KEYS = [
    "pickup_latitude", "pickup_longitude", "dropoff_latitude",
    "dropoff_longitude"
]

CATEGORICAL_FEATURE_KEYS = [
    "pickup_census_tract", "dropoff_census_tract", "pickup_community_area",
    "dropoff_community_area", "trip_start_hour", "trip_start_day", "trip_start_month"
]

LABEL_KEY = "tips"

FARE_KEY = "fare"
