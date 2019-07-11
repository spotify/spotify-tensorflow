#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
import numpy as np
import tensorflow as tf
from spotify_tensorflow.tfx.model_analysis import fpl_utils
from tests.model_analysis.test_post_processing_extractor import PostProcessingExtractorTest


class FplUtilsTest(tf.test.TestCase):
    def test_getters(self):
        fpls = PostProcessingExtractorTest.create_fpls()
        self.assertAllEqual(np.array(["f"]), fpl_utils.get_feature_value(fpls[0], "gender"))
        self.assertAllEqual(np.array(["m"]), fpl_utils.get_feature_value(fpls[1], "gender"))
        self.assertAllEqual(np.array([13]), fpl_utils.get_feature_value(fpls[0], "age"))
        self.assertAllEqual(np.array([10]), fpl_utils.get_feature_value(fpls[1], "age"))
        self.assertAllEqual(np.array(["cars"]), fpl_utils.get_feature_value(fpls[0], "interest"))
        self.assertAllEqual(
            np.array(["cars", "movies"]), fpl_utils.get_feature_value(fpls[1], "interest")
        )

        self.assertAllEqual(
            np.array([[1, 0]]), fpl_utils.get_prediction_value(fpls[0], "prediction_key0")
        )
        self.assertAllEqual(
            np.array([[1, 0]]), fpl_utils.get_prediction_value(fpls[1], "prediction_key0")
        )
        self.assertAllEqual(
            np.array([["B", "C"]]), fpl_utils.get_prediction_value(fpls[0], "prediction_key1")
        )
        self.assertAllEqual(
            np.array([["B", "C"]]), fpl_utils.get_prediction_value(fpls[1], "prediction_key1")
        )

        self.assertEqual(0, fpl_utils.get_label_value(fpls[0], "label_key0"))
        self.assertEqual(1, fpl_utils.get_label_value(fpls[1], "label_key0"))
        self.assertAllEqual(np.array(["A", "B"]), fpl_utils.get_label_value(fpls[0], "label_key1"))
        self.assertAllEqual(np.array(["A", "B"]), fpl_utils.get_label_value(fpls[1], "label_key1"))

    def test_setters(self):
        fpls = PostProcessingExtractorTest.create_fpls()
        self.assertListEqual(
            [0],
            list(
                fpl_utils.get_feature_value(
                    fpl_utils.set_feature(fpls[0], "gender", [0]), "gender"
                )
            ),
        )
        self.assertListEqual(
            ["other"],
            list(
                fpl_utils.get_feature_value(
                    fpl_utils.set_feature(fpls[1], "gender", ["other"]), "gender"
                )
            ),
        )
        self.assertListEqual(
            ["a", "b"],
            list(
                fpl_utils.get_feature_value(
                    fpl_utils.set_feature(fpls[0], "age", ["a", "b"]), "age"
                )
            ),
        )
        self.assertListEqual(
            [54],
            list(fpl_utils.get_feature_value(fpl_utils.set_feature(fpls[1], "age", 54), "age")),
        )
        self.assertListEqual(
            [],
            list(
                fpl_utils.get_feature_value(
                    fpl_utils.set_feature(fpls[0], "interest", []), "interest"
                )
            ),
        )
        self.assertListEqual(
            ["cars", "cars"],
            list(
                fpl_utils.get_feature_value(
                    fpl_utils.set_feature(fpls[1], "interest", ["cars", "cars"]), "interest"
                )
            ),
        )

        self.assertTrue(
            np.all(
                np.array([[0, 1], [2, 3]])
                == fpl_utils.get_prediction_value(
                    fpl_utils.set_prediction(fpls[0], "prediction_key0", [[0, 1], [2, 3]]),
                    "prediction_key0",
                )
            )
        )
        self.assertTrue(
            np.all(
                np.array([[0], [2, 3]])
                == fpl_utils.get_label_value(
                    fpl_utils.set_label(fpls[0], "label_key0", [[0], [2, 3]]), "label_key0"
                )
            )
        )
