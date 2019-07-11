#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
"""
TFMA uses heavily the type FeaturePredictionLabels (FPL) between extracts.
It's defined as:
FeaturesPredictionsLabels = NamedTuple(
    'FeaturesPredictionsLabels', [('input_ref', int),
                                  ('features', DictOfFetchedTensorValues),
                                  ('predictions', DictOfFetchedTensorValues),
                                  ('labels', DictOfFetchedTensorValues)])
These methods are meant to help deal with it.

"""
import copy

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import types, constants
from tensorflow_model_analysis.eval_saved_model import encoding


def get_feature_value(fpl, feature_key):
    """Helper to get value from FPL dict."""
    node_value = fpl.features[feature_key][encoding.NODE_SUFFIX]
    if isinstance(node_value, tf.compat.v1.SparseTensorValue):
        return node_value.values
    return node_value


def get_label_value(fpl, label_key):
    """Helper to get value from FPL dict."""
    node_value = fpl.labels[label_key][encoding.NODE_SUFFIX]
    if isinstance(node_value, tf.compat.v1.SparseTensorValue):
        return node_value.values
    return node_value


def get_prediction_value(fpl, prediction_key):
    """Helper to get value from FPL dict."""
    node_value = fpl.predictions[prediction_key][encoding.NODE_SUFFIX]
    if isinstance(node_value, tf.compat.v1.SparseTensorValue):
        return node_value.values
    return node_value


def _set_key_value(fpl_member, key, value):
    """Helper to set feature in FPL dict."""
    if isinstance(value, list):
        value = np.array(value)
    if not isinstance(value, np.ndarray) and not isinstance(value, tf.compat.v1.SparseTensorValue):
        value = np.array([value])
    fpl_member[key] = {encoding.NODE_SUFFIX: value}
    return fpl_member  # pytype: disable=bad-return-type


def set_feature(fpl, feature_key, feature_value):
    fpl_copy = types.FeaturesPredictionsLabels(
        features=_set_key_value(fpl.features, feature_key, feature_value),
        predictions=fpl.predictions,
        labels=fpl.labels,
        input_ref=fpl.input_ref,
    )
    return fpl_copy


def set_prediction(fpl, prediction_key, prediction_value):
    fpl_copy = types.FeaturesPredictionsLabels(
        features=fpl.features,
        predictions=_set_key_value(fpl.predictions, prediction_key, prediction_value),
        labels=fpl.labels,
        input_ref=fpl.input_ref,
    )
    return fpl_copy


def set_label(fpl, label_key, label_value):
    fpl_copy = types.FeaturesPredictionsLabels(
        features=fpl.features,
        predictions=fpl.predictions,
        labels=_set_key_value(fpl.labels, label_key, label_value),
        input_ref=fpl.input_ref,
    )
    return fpl_copy


def get_fpl_copy(fpl):
    if not fpl:
        raise RuntimeError("FPL missing, Please ensure _Predict() was called.")

    # We must make a copy of the FPL tuple as well, so that we don't mutate the
    # original which is disallowed by Beam.
    fpl_copy = types.FeaturesPredictionsLabels(
        features=copy.copy(fpl.features),
        labels=copy.copy(fpl.labels),
        predictions=copy.copy(fpl.predictions),
        input_ref=fpl.input_ref,
    )
    return fpl_copy


def get_fpl_copy_from_extracts(extracts):
    """Get a copy of the FPL in the extracts of extracts."""
    fpl_orig = extracts.get(constants.FEATURES_PREDICTIONS_LABELS_KEY)
    return get_fpl_copy(fpl_orig)
