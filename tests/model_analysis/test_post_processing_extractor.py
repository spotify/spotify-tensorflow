#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB

"""Test for using the _PostProcessing extractors part of TFMA."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
import numpy as np
import tensorflow as tf
from apache_beam.testing import util
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import testutil
from spotify_tensorflow.tfx.model_analysis import post_processing_extractor, fpl_utils
from spotify_tensorflow.tfx.model_analysis.post_processing_fns import \
    remove_feature_from_predictions


class PostProcessingExtractorTest(testutil.TensorflowModelAnalysisTest):
    @staticmethod
    def make_features_dict(features_dict):
        result = {}
        for key, value in features_dict.items():
            result[key] = {"node": np.array(value)}
        return result

    @staticmethod
    def create_fpls():
        """Create test FPL dicts that can be used for verification."""
        fpl1 = types.FeaturesPredictionsLabels(
            input_ref=0,
            features=PostProcessingExtractorTest.make_features_dict(
                {"gender": ["f"], "age": [13], "interest": ["cars"], "feature_key1": ["B"]}
            ),
            predictions=PostProcessingExtractorTest.make_features_dict(
                {"prediction_key0": [np.array([1, 0])], "prediction_key1": [np.array(["B", "C"])]}
            ),
            labels=PostProcessingExtractorTest.make_features_dict(
                {"label_key0": 0, "label_key1": ["A", "B"]}
            ),
        )
        fpl2 = types.FeaturesPredictionsLabels(
            input_ref=1,
            features=PostProcessingExtractorTest.make_features_dict(
                {
                    "gender": ["m"],
                    "age": [10],
                    "interest": ["cars", "movies"],
                    "feature_key1": ["B"],
                }
            ),
            predictions=PostProcessingExtractorTest.make_features_dict(
                {"prediction_key0": [np.array([1, 0])], "prediction_key1": [np.array(["B", "C"])]}
            ),
            labels=PostProcessingExtractorTest.make_features_dict(
                {"label_key0": 1, "label_key1": ["A", "B"]}
            ),
        )
        return [fpl1, fpl2]

    def testBadReturnType(self):
        def bad_post_processing_fn(_):
            return {"bad": ["type"]}

        with self.assertRaises(TypeError):
            with beam.Pipeline() as pipeline:
                fpls = self.create_fpls()

                (
                    pipeline
                    | "CreateTestInput" >> beam.Create(fpls)
                    | "WrapFpls" >> beam.Map(wrap_test_fpl)
                    | "RemoveLabelsFromPredictions"
                    >> post_processing_extractor._PostProcessing(bad_post_processing_fn)
                )

    # def testPicklable(self):
    #     def no_op_fn(fpl):
    #         return fpl
    #
    #     def check_result(got):
    #         self.assertListEqual(None, got)
    #
    #     pickler.loads(pickler.dumps(post_processing_extractor._PostProcessing(no_op_fn)))
    #
    #     # Next one does not work on python3.6 and pytest
    #     pickler.loads(pickler.dumps(beam.Map(check_result)))

    def testDummyPipeline(self):
        with beam.Pipeline() as pipeline:
            fpls = ["a", "b"]

            output_fpls = pipeline | "CreateTestInput" >> beam.Create(fpls)

        def check_result(got):
            assert fpls == got

        util.assert_that(output_fpls, check_result)

    def testNoOp(self):
        def no_op_fn(fpl):
            return fpl

        with beam.Pipeline() as pipeline:
            fpls = self.create_fpls()

            outputpost_preocessed_fpls = (
                pipeline
                | "CreateTestInput" >> beam.Create(fpls)
                | "WrapFpls" >> beam.Map(wrap_test_fpl)
                | "RemoveLabelsFromPredictions"
                >> post_processing_extractor._PostProcessing(no_op_fn)
            )

        def check_result(got):
            assert len(fpls) == len(got)
            assert fpls == got

        util.assert_that(outputpost_preocessed_fpls, check_result)

    def testPostProcessing_remove_feature_from_predictions(self):
        with beam.Pipeline() as pipeline:
            fpls = self.create_fpls()

            new_fpls = (
                pipeline
                | "CreateTestInput" >> beam.Create(fpls)
                | "WrapFpls" >> beam.Map(wrap_test_fpl)
                | "RemoveLabelsFromPredictions"
                >> post_processing_extractor._PostProcessing(
                    remove_feature_from_predictions("feature_key1", "prediction_key1")
                )
            )

            def check_result(got):
                try:
                    assert 2 == len(got)
                    for res in got:
                        assert ["C"] == list(
                            fpl_utils.get_prediction_value(
                                res[constants.FEATURES_PREDICTIONS_LABELS_KEY], "prediction_key1"
                            )
                        )
                except AssertionError as err:
                    raise util.BeamAssertException(err)

            util.assert_that(new_fpls, check_result)


def wrap_test_fpl(fpl):
    return {constants.INPUT_KEY: "xyz", constants.FEATURES_PREDICTIONS_LABELS_KEY: fpl}
