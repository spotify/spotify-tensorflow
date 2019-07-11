#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
import numpy as np
import tensorflow as tf
from spotify_tensorflow.tfx.model_analysis import post_processing_fns, fpl_utils
from tests.model_analysis.test_post_processing_extractor import PostProcessingExtractorTest
from tensorflow_model_analysis import types


class PostProcessingFnsTest(tf.test.TestCase):
    @staticmethod
    def create_fpls():
        """Create test FPL dicts that can be used for verification."""
        fpl0 = types.FeaturesPredictionsLabels(
            input_ref=0,
            features=PostProcessingExtractorTest.make_features_dict({"feature_key1": []}),
            predictions=PostProcessingExtractorTest.make_features_dict(
                {"prediction_key0": [np.array([1, 0])], "prediction_key1": [np.array(["B", "C"])]}
            ),
            labels=PostProcessingExtractorTest.make_features_dict(
                {"label_key0": 0, "label_key1": ["A", "B"]}
            ),
        )
        fpl1 = types.FeaturesPredictionsLabels(
            input_ref=0,
            features=PostProcessingExtractorTest.make_features_dict({"feature_key1": ["B"]}),
            predictions=PostProcessingExtractorTest.make_features_dict(
                {"prediction_key0": [np.array([1, 0])], "prediction_key1": [np.array(["B", "C"])]}
            ),
            labels=PostProcessingExtractorTest.make_features_dict(
                {"label_key0": 0, "label_key1": ["A", "B"]}
            ),
        )
        fpl2 = types.FeaturesPredictionsLabels(
            input_ref=1,
            features=PostProcessingExtractorTest.make_features_dict({"feature_key1": ["B", "C"]}),
            predictions=PostProcessingExtractorTest.make_features_dict(
                {"prediction_key0": [np.array([0, 1])], "prediction_key1": [np.array(["B", "C"])]}
            ),
            labels=PostProcessingExtractorTest.make_features_dict(
                {"label_key0": 1, "label_key1": ["A", "B"]}
            ),
        )
        return [fpl0, fpl1, fpl2]

    def test_remove_labels_from_predictions(self):
        fpls = self.create_fpls()
        prediction_key = "prediction_key1"
        fn = post_processing_fns.remove_feature_from_predictions("feature_key1", "prediction_key1")
        new_fpls = [fn(fpl) for fpl in fpls]

        def get_prediction_value(fpl, key):
            return list(fpl_utils.get_prediction_value(fpl, key))

        # Nothing to remove
        self.assertAllEqual([["B", "C"]], get_prediction_value(new_fpls[0], prediction_key))
        self.assertAllEqual([[1, 0]], get_prediction_value(new_fpls[0], "prediction_key0"))

        # One thing to remove
        self.assertAllEqual([["C"]], get_prediction_value(new_fpls[1], prediction_key))
        self.assertAllEqual([[0]], get_prediction_value(new_fpls[1], "prediction_key0"))

        # Everything to remove
        self.assertAllEqual([[]], get_prediction_value(new_fpls[2], prediction_key))
        self.assertAllEqual([[]], get_prediction_value(new_fpls[2], "prediction_key0"))
