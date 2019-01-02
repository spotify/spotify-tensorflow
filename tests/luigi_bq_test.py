# -*- coding: utf-8 -*-
#
#  Copyright 2017 Spotify AB.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

from __future__ import absolute_import, division, print_function

from datetime import date

from spotify_tensorflow.luigi.bq import BigQueryDailySnapshot, table_str
from tensorflow.python.platform import test


class LuigiBqTest(test.TestCase):

    @staticmethod
    def test_bigquery_daily_snapshot_normalize_table_name():

        cases = [
            ("table_1_suf", ("", "table_1", "suf")),
            ("table_1_suf", ("dataset_1", "table_1", "suf")),
            ("dataset_1_suf", ("dataset_1", "", "suf"))
        ]

        for expected, (dataset, table, suffix) in cases:
            table_name = BigQueryDailySnapshot._normalize_table_name(dataset, table, suffix)
            assert expected == table_name, "({}, {}, {}) does not normalize to {}; got {}".format(
                dataset, table, suffix, expected, table_name)

        # Empty case should raise
        try:
            BigQueryDailySnapshot._normalize_table_name("", "", "")
            assert False
        except ValueError:
            assert True

    @staticmethod
    def test_biquery_daily_snapshot_raises_on_empty_dataset():
        try:
            BigQueryDailySnapshot(project="some-project", dataset="",
                                  table="", date=date(2808, 1, 2))
            assert False
        except ValueError:
            assert True

    @staticmethod
    def test_table_str():
        cases = [
            ("some-project.dataset_1.table_1_28080102", ("dataset_1", "table_1", date(2808, 1, 2))),
            ("some-project.dataset_1.dataset_1_28080102", ("dataset_1", "", date(2808, 1, 2)))
        ]

        for expected, (dataset, table, dt) in cases:
            answer = table_str(BigQueryDailySnapshot(project="some-project", dataset=dataset,
                                                     table=table, date=dt).output())
            assert expected == answer, "({}, {}, {}) does not normalize to {}; got {}".format(
                dataset, table, str(dt), expected, answer)
