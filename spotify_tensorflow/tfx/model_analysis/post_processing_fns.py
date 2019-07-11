#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
import numpy as np
from spotify_tensorflow.tfx.model_analysis import fpl_utils


def remove_feature_from_predictions(feature_key, prediction_key):
    """Remove elements from all members of fpl.predictions at indices of
    fpl.predictions[prediction_key] that are present
    in fpl.features[label_key]. Order is preserved.

    Args:
      fpl: One mutable instance of FeaturePredictionLabels to be modified
      feature_key: key into the feature dictionary of fpl to use as a reference of element to
      exclude from predictions
      prediction_key: key into the predictions dictionary of the fpl to be modified

    Returns:
      the modified FeaturePredictionLabels
    """

    def make_new_predictions_dict(fpl, features_to_remove, reference_predictions):
        # features_to_remove is a ndarray of 1 or 2 dimension. It does not matter because __in__
        # is able to handle both cases
        new_predictions = dict()
        all_prediction_keys = fpl.predictions.keys()
        old_flat_predictions = {
            key: np.reshape(fpl_utils.get_prediction_value(fpl, key), -1)
            for key in all_prediction_keys
        }
        for key in all_prediction_keys:
            new_predictions[key] = []

        for k, reference_prediction in enumerate(reference_predictions):
            if reference_prediction not in features_to_remove:
                for key in all_prediction_keys:
                    new_predictions[key].append(old_flat_predictions[key][k])
        for key in all_prediction_keys:
            new_predictions[key] = np.array(new_predictions[key]).reshape((1, -1))
        return new_predictions

    def do(fpl):
        # TODO: make assertion on the predictions shapes and maybe raise an error
        features_to_remove = fpl_utils.get_feature_value(fpl, feature_key)
        # predictions are of shape [batch_size, ?] but the batch_size is always 1 in this case
        reference_predictions = np.reshape(fpl_utils.get_prediction_value(fpl, prediction_key), -1)
        new_predictions = make_new_predictions_dict(fpl, features_to_remove, reference_predictions)

        fpl_copy = fpl_utils.get_fpl_copy(fpl)
        for key, value in new_predictions.items():
            fpl_copy = fpl_utils.set_prediction(fpl_copy, key, value)
        return fpl_copy

    return do
