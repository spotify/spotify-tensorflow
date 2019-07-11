#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
import tensorflow as tf


def get_filter_input_example_fn(filtering_config):
    """
    Returns a preprocessing_fn that can be used in the PreProcessingExtractor.

    filtering_config:
        dictionary of feature_name to the number of element to keep.
        For all feature in the example that are not in filtering_config, all the values are kept.

    """

    def filter_input_example(example):
        """Depending on the filtering_config, will keep only the first nb_items_to_keep if provided

        example: tf.train.Example
        """
        new_features = {}
        for feature_name, feature in example.features.feature.items():
            if feature_name in filtering_config:
                new_features[feature_name] = _filter_feature(
                    feature, nb_items_to_keep=filtering_config[feature_name]
                )
            else:
                new_features[feature_name] = feature

        new_tf_example = tf.train.Example(features=tf.train.Features(feature=new_features))
        return new_tf_example

    return filter_input_example


def _filter_feature(feature, nb_items_to_keep):
    if not nb_items_to_keep:
        return feature

    if feature.bytes_list.value:
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=feature.bytes_list.value[:nb_items_to_keep])
        )

    elif feature.float_list.value:
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=feature.float_list.value[:nb_items_to_keep])
        )

    elif feature.int64_list.value:
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=feature.int64_list.value[:nb_items_to_keep])
        )
