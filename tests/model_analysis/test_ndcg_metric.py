#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
import random
import string

import tensorflow as tf
import numpy as np
from spotify_tensorflow.tfx.model_analysis.ndcg_post_export_metrics import dcg, \
    compute_batch_updates


class TestNdcgMetric(tf.test.TestCase):
    SIZE_ID = 10

    def get_idcg(self, length):
        return self.dcg_decay(length - 1) + 1

    def dcg_decay(self, length, offset=1):
        return np.sum(1 / np.log2(1 + np.arange(offset + 1, offset + length + 1)))

    def retrieved_from_sublist(self, expected_r, relevant_ids):
        relevant_len = len(relevant_ids)
        retids = random.sample(relevant_ids, int(expected_r * relevant_len))
        random.shuffle(retids)
        return retids

    def retrieved_with_errors_bottom(self, expected_r, relevant_ids):
        retids = self.retrieved_from_sublist(expected_r, relevant_ids)
        retids.extend(self.random_strings(self.SIZE_ID, 100))
        return retids

    def retrieved_with_errors_top(self, expected_r, relevant_ids, offset=100):
        retids = self.random_strings(self.SIZE_ID, offset)
        retids.extend(self.retrieved_from_sublist(expected_r, relevant_ids))
        return retids

    def random_string(self, size, chars=string.ascii_uppercase + string.digits):
        return "".join(random.choice(chars) for _ in range(size))

    def random_strings(self, size, length, chars=string.ascii_uppercase + string.digits):
        return [self.random_string(size, chars) for _ in range(length)]

    def ndcg(self, retrieved_elements, relevant_elements, k):
        idcg = dcg(relevant_elements[:k], relevant_elements[:k])
        true_dcg = dcg(retrieved_elements[:k], relevant_elements[:k])

        return true_dcg / idcg

    def test_ndcg(self):
        rel_length = 100

        rel_ids = self.random_strings(self.SIZE_ID, rel_length)

        this_idcg = self.get_idcg(rel_length)
        assert dcg(rel_ids, rel_ids) == this_idcg

        tests = [
            (rel_ids, 1),
            (self.retrieved_from_sublist(0.01, rel_ids), self.get_idcg(1) / this_idcg),
            (self.retrieved_from_sublist(0.1, rel_ids), self.get_idcg(10) / this_idcg),
            (self.retrieved_from_sublist(0.9, rel_ids), self.get_idcg(90) / this_idcg),
            (self.retrieved_from_sublist(1, rel_ids), 1),
            # Errors on bottom
            (self.retrieved_with_errors_bottom(0, rel_ids), 0),
            (self.retrieved_with_errors_bottom(0.1, rel_ids), self.get_idcg(10) / this_idcg),
            (self.retrieved_with_errors_bottom(1, rel_ids), 1),
            # Errors on top
            (self.retrieved_with_errors_top(0, rel_ids), 0),
            (self.retrieved_with_errors_top(0.1, rel_ids), 0),
            (self.retrieved_with_errors_top(1, rel_ids), 0),
        ]

        for retrieved, expected_ndcg in tests:
            self.assertAllClose(self.ndcg(retrieved, rel_ids, rel_length), expected_ndcg)

        # Test with a bigger K
        self.assertEqual(
            self.ndcg(self.retrieved_with_errors_top(0, rel_ids), rel_ids, k=2 * rel_length), 0
        )
        self.assertAllClose(
            self.ndcg(self.retrieved_with_errors_top(0.1, rel_ids), rel_ids, k=2 * rel_length),
            self.dcg_decay(10, 100) / this_idcg,
        )
        self.assertAllClose(
            self.ndcg(self.retrieved_with_errors_top(1, rel_ids), rel_ids, k=2 * rel_length),
            self.dcg_decay(100, 100) / this_idcg,
        )

    def test_compute_batch_updates(self):
        classes = np.array([["a", "b", "c"],
                            ["a", "b", "c"],
                            ["a", "b", "c"],
                            ])
        scores = np.array([[0.9, 0.8, 0.7],
                           [0.9, 0.2, 0.1],
                           [0.1, 0.2, 0.9],
                           ])
        labels = np.array([["a", "c", ""],
                           ["a", "", ""],
                           ["a", "", ""],
                           ])
        weights = np.ones((3, 1))
        ndcg_tensor, total_weight_tensor = compute_batch_updates(classes, scores, labels, weights,
                                                                 cutoffs=[1, 3])
        self.assertAllClose(np.array([2 / 3., 0.8065735963827292]),
                            ndcg_tensor / total_weight_tensor)
