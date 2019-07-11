#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import (
    fixed_prediction_classifier_extra_fields,
)
from tensorflow_model_analysis.extractors import predict_extractor, slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.slicer import slicer
from spotify_tensorflow.tfx.model_analysis import post_processing_extractor, post_processing_fns, \
    pre_processing_fns, pre_processing_extractor, ndcg_post_export_metrics


class IntegrationWithHighLevelTfmaTest(testutil.TensorflowModelAnalysisTest):
    # Stolen from tfma.api.model_eval_lib_test.testRunModelAnalysisExtraFieldsPlusFeatureExtraction

    def setUp(self):
        super(IntegrationWithHighLevelTfmaTest, self).setUp()
        self.model_location = self._export_eval_saved_model(
            fixed_prediction_classifier_extra_fields.simple_fixed_prediction_classifier_extra_fields
        )
        examples = [
            self._makeExample(
                classes=["a", "b", "c"],
                scores=[0.9, 0.8, 0.7],
                labels=["a", "c"],
                fixed_float=1.0,
                fixed_string="a",
                fixed_int=0,
            ),
            self._makeExample(
                classes=["a", "b", "c"],
                scores=[0.9, 0.2, 0.1],
                labels=["a"],
                fixed_float=2.0,
                fixed_string="a",
                fixed_int=0,
            ),
            self._makeExample(
                classes=["a", "b", "c"],
                scores=[0.1, 0.2, 0.9],
                labels=["a"],
                fixed_float=3.0,
                fixed_string="b",
                fixed_int=0,
            ),
        ]
        self.data_location = self._write_tfexamples_to_tfrecords(examples)
        self.slice_spec = [slicer.SingleSliceSpec(columns=["fixed_string"])]

    @staticmethod
    def make_expected_post_metric(metric_key, cutoffs, values):
        value_dicts = []
        for cutoff, value in zip(cutoffs, values):
            value_dict = dict(cutoff=cutoff, boundedValue=dict(value=float(value)))
            if value > 0.0:
                value_dict["value"] = float(value)
            value_dicts.append(value_dict)
        return {metric_key: dict(valueAtCutoffs=dict(values=value_dicts))}

    def _export_eval_saved_model(self, classifier):
        temp_eval_export_dir = os.path.join(self.get_temp_dir(), "eval_export_dir")
        _, eval_export_dir = classifier(None, temp_eval_export_dir)
        return eval_export_dir

    def _write_tfexamples_to_tfrecords(self, examples):
        data_location = os.path.join(self.get_temp_dir(), "input_data.rio")
        with tf.io.TFRecordWriter(data_location) as writer:
            for example in examples:
                writer.write(example.SerializeToString())
        return data_location

    def assert_metrics_almost_equal(self, got_value, expected_value):
        if got_value:
            for (s, m) in got_value:
                self.assertIn(s, expected_value)
                for k in expected_value[s]:
                    self.assertIn(k, m)
                    self.assertDictElementsAlmostEqual(m[k], expected_value[s][k])
        else:
            # Only pass if expected_value also evaluates to False.
            self.assertFalse(expected_value, msg="Actual value was empty.")

    def assert_post_metrics_almost_equals(self, metric_key, eval_result, expected_post_metrics):
        for slice_key, metric in eval_result.slicing_metrics:
            values = metric[metric_key]["valueAtCutoffs"]["values"]
            expected_values = expected_post_metrics[slice_key][metric_key]["valueAtCutoffs"][
                "values"
            ]
            for value_dict, expected_value_dict in zip(values, expected_values):
                for key, expected_value in expected_value_dict.items():
                    self.assertIn(key, value_dict)
                    if not isinstance(value_dict[key], dict):
                        self.assertAlmostEqual(value_dict[key], expected_value, places=5)

    def test_basic_tfma_usage(self):
        eval_shared_model = model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=self.model_location, example_weight_key="fixed_float"
        )
        default_extractors = [
            predict_extractor.PredictExtractor(
                eval_shared_model, desired_batch_size=3, materialize=False
            ),
            slice_key_extractor.SliceKeyExtractor(self.slice_spec, materialize=False),
        ]
        default_evaluators = tfma.default_evaluators(eval_shared_model)
        eval_result = model_eval_lib.run_model_analysis(
            eval_shared_model,
            self.data_location,
            slice_spec=self.slice_spec,
            extractors=default_extractors,
            evaluators=default_evaluators,
        )
        # We only check some of the metrics to ensure that the end-to-end
        # pipeline works.
        expected = {
            (("fixed_string", b"a"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 2.0},
            },
            (("fixed_string", b"b"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 1.0},
            },
        }
        self.assertEqual(eval_result.config.model_location, self.model_location)
        self.assertEqual(eval_result.config.data_location, self.data_location)
        self.assertEqual(eval_result.config.slice_spec, self.slice_spec)
        self.assert_metrics_almost_equal(eval_result.slicing_metrics, expected)
        self.assertFalse(eval_result.plots)

    def test_integration_with_post_metrics(self):
        post_metrics = [tfma.post_export_metrics.precision_at_k([1, 10])]
        eval_shared_model = model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=self.model_location,
            example_weight_key="fixed_float",
            add_metrics_callbacks=post_metrics,
        )
        default_extractors = [
            predict_extractor.PredictExtractor(
                eval_shared_model, desired_batch_size=3, materialize=False
            ),
            slice_key_extractor.SliceKeyExtractor(self.slice_spec, materialize=False),
        ]
        default_evaluators = tfma.default_evaluators(eval_shared_model)

        eval_result = model_eval_lib.run_model_analysis(
            eval_shared_model,
            self.data_location,
            slice_spec=self.slice_spec,
            extractors=default_extractors,
            evaluators=default_evaluators,
        )
        # We only check some of the metrics to ensure that the end-to-end
        # pipeline works.
        standard_expected_metrics = {
            (("fixed_string", b"a"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 2.0},
            },
            (("fixed_string", b"b"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 1.0},
            },
        }
        expected_post_export_metrics = {
            (("fixed_string", b"a"),): {
                metric_keys.PRECISION_AT_K: {
                    "valueAtCutoffs": {
                        "values": [
                            {"cutoff": 1, "boundedValue": {"value": 1.0}, "value": 1.0},
                            {"cutoff": 10, "boundedValue": {"value": 0.5}, "value": 0.5},
                        ]
                    }
                }
            },
            (("fixed_string", b"b"),): {
                metric_keys.PRECISION_AT_K: {
                    "valueAtCutoffs": {
                        "values": [
                            {"cutoff": 1, "boundedValue": {"value": 0.0}},
                            {
                                "cutoff": 10,
                                "boundedValue": {"value": 1.0 / 3.0},
                                "value": 1.0 / 3.0,
                            },
                        ]
                    }
                }
            },
        }
        self.assertEqual(eval_result.config.model_location, self.model_location)
        self.assertEqual(eval_result.config.data_location, self.data_location)
        self.assertEqual(eval_result.config.slice_spec, self.slice_spec)
        self.assert_metrics_almost_equal(eval_result.slicing_metrics, standard_expected_metrics)
        self.assert_post_metrics_almost_equals(
            metric_keys.PRECISION_AT_K, eval_result, expected_post_export_metrics
        )
        self.assertFalse(eval_result.plots)

    def test_integration_with_post_processing(self):
        post_metrics = [tfma.post_export_metrics.precision_at_k([1, 10])]
        eval_shared_model = model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=self.model_location,
            example_weight_key="fixed_float",
            add_metrics_callbacks=post_metrics,
        )
        extractors = [
            predict_extractor.PredictExtractor(
                eval_shared_model, desired_batch_size=3, materialize=False
            ),
            slice_key_extractor.SliceKeyExtractor(self.slice_spec, materialize=False),
            # Remove fixed_string form classes
            post_processing_extractor.PostProcessingExtractor(
                post_processing_fns.remove_feature_from_predictions("fixed_string", "classes")
            ),
        ]
        evaluators = [
            tfma.evaluators.MetricsAndPlotsEvaluator(
                eval_shared_model, run_after=tfma.extractors.LAST_EXTRACTOR_STAGE_NAME
            )
        ]

        eval_result = model_eval_lib.run_model_analysis(
            eval_shared_model,
            self.data_location,
            slice_spec=self.slice_spec,
            extractors=extractors,
            evaluators=evaluators,
        )
        # We only check some of the metrics to ensure that the end-to-end
        # pipeline works.
        standard_expected_metrics = {
            (("fixed_string", b"a"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 2.0},
            },
            (("fixed_string", b"b"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 1.0},
            },
        }
        expected_post_export_metrics = {
            (("fixed_string", b"a"),): self.make_expected_post_metric(
                metric_keys.PRECISION_AT_K, [1, 10], [0.0, 0.25]
            ),
            (("fixed_string", b"b"),): self.make_expected_post_metric(
                metric_keys.PRECISION_AT_K, [1, 10], [0.0, 0.5]
            ),
        }
        self.assertEqual(eval_result.config.model_location, self.model_location)
        self.assertEqual(eval_result.config.data_location, self.data_location)
        self.assertEqual(eval_result.config.slice_spec, self.slice_spec)
        self.assert_metrics_almost_equal(eval_result.slicing_metrics, standard_expected_metrics)
        self.assert_post_metrics_almost_equals(
            metric_keys.PRECISION_AT_K, eval_result, expected_post_export_metrics
        )
        self.assertFalse(eval_result.plots)

    def test_integration_with_pre_processing(self):
        post_metrics = [tfma.post_export_metrics.precision_at_k([1, 10])]
        eval_shared_model = model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=self.model_location,
            example_weight_key="fixed_float",
            add_metrics_callbacks=post_metrics,
        )
        filtering_config = {"classes": 2, "fixed_int": 3, "labels": 1}
        extractors = [
            pre_processing_extractor.PreProcessingExtractor(
                pre_processing_fns.get_filter_input_example_fn(filtering_config)
            ),
            predict_extractor.PredictExtractor(
                eval_shared_model, desired_batch_size=3, materialize=False
            ),
            slice_key_extractor.SliceKeyExtractor(self.slice_spec, materialize=False),
            # Remove fixed_string form classes
            post_processing_extractor.PostProcessingExtractor(
                post_processing_fns.remove_feature_from_predictions("fixed_string", "classes")
            ),
        ]
        evaluators = [
            tfma.evaluators.MetricsAndPlotsEvaluator(
                eval_shared_model, run_after=tfma.extractors.LAST_EXTRACTOR_STAGE_NAME
            )
        ]

        eval_result = model_eval_lib.run_model_analysis(
            eval_shared_model,
            self.data_location,
            extractors=extractors,
            evaluators=evaluators,
            slice_spec=self.slice_spec,
        )
        # We only check some of the metrics to ensure that the end-to-end
        # pipeline works.
        standard_expected_metrics = {
            (("fixed_string", b"a"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 2.0},
            },
            (("fixed_string", b"b"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 1.0},
            },
        }
        expected_post_export_metrics = {
            (("fixed_string", b"a"),): self.make_expected_post_metric(
                metric_keys.PRECISION_AT_K, [1, 10], [0.0, 0.0]
            ),
            (("fixed_string", b"b"),): self.make_expected_post_metric(
                metric_keys.PRECISION_AT_K, [1, 10], [1.0, 1.0]
            ),
        }
        self.assertEqual(eval_result.config.model_location, self.model_location)
        self.assertEqual(eval_result.config.data_location, self.data_location)
        self.assertEqual(eval_result.config.slice_spec, self.slice_spec)
        self.assert_metrics_almost_equal(eval_result.slicing_metrics, standard_expected_metrics)
        self.assert_post_metrics_almost_equals(
            metric_keys.PRECISION_AT_K, eval_result, expected_post_export_metrics
        )
        self.assertFalse(eval_result.plots)

    def test_integration_with_ndcg(self):
        post_metrics = [
            tfma.post_export_metrics.precision_at_k([1, 10]),
            ndcg_post_export_metrics.ndcg_at_k([1, 3]),
        ]
        eval_shared_model = model_eval_lib.default_eval_shared_model(
            eval_saved_model_path=self.model_location,
            example_weight_key="fixed_float",
            add_metrics_callbacks=post_metrics,
        )
        extractors = [
            predict_extractor.PredictExtractor(
                eval_shared_model, desired_batch_size=3, materialize=False
            ),
            slice_key_extractor.SliceKeyExtractor(self.slice_spec, materialize=False),
        ]

        default_evaluators = tfma.default_evaluators(eval_shared_model)

        eval_result = model_eval_lib.run_model_analysis(
            eval_shared_model,
            self.data_location,
            slice_spec=self.slice_spec,
            extractors=extractors,
            evaluators=default_evaluators,
        )
        # We only check some of the metrics to ensure that the end-to-end
        # pipeline works.
        standard_expected_metrics = {
            (("fixed_string", b"a"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 2.0},
            },
            (("fixed_string", b"b"),): {
                metric_keys.EXAMPLE_WEIGHT: {"doubleValue": 3.0},
                metric_keys.EXAMPLE_COUNT: {"doubleValue": 1.0},
            },
        }
        expected_post_export_metrics = {
            (("fixed_string", b"a"),): self.make_expected_post_metric(
                ndcg_post_export_metrics.NDCG_AT_K, [1, 3], [1.0, 0.959860394574]
            ),
            (("fixed_string", b"b"),): self.make_expected_post_metric(
                ndcg_post_export_metrics.NDCG_AT_K, [1, 3], [0.0, 0.5]
            ),
        }
        self.assertEqual(eval_result.config.model_location, self.model_location)
        self.assertEqual(eval_result.config.data_location, self.data_location)
        self.assertEqual(eval_result.config.slice_spec, self.slice_spec)
        self.assert_metrics_almost_equal(eval_result.slicing_metrics, standard_expected_metrics)
        self.assert_post_metrics_almost_equals(
            ndcg_post_export_metrics.NDCG_AT_K, eval_result, expected_post_export_metrics
        )
        self.assertFalse(eval_result.plots)


if __name__ == "__main__":
    tf.test.main()
