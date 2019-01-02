# -*- coding: utf-8 -*-
#
# Copyright 2017 Spotify AB.
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

from __future__ import absolute_import, division, print_function

import luigi
from luigi.contrib.bigquery import BigqueryTarget
from spotify_tensorflow.luigi.external_daily_snapshot import ExternalDailySnapshot


class BigQueryDailySnapshot(ExternalDailySnapshot):
    project = luigi.Parameter()
    dataset = luigi.Parameter()
    table = luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        super(ExternalDailySnapshot, self).__init__(*args, **kwargs)
        if self.dataset == "":
            raise ValueError("non-empty dataset parameter must be provided.")

    def output(self):
        table_name = self._normalize_table_name(self.dataset, self.table,
                                                suffix=self._sanitize_date(self.date))
        return BigqueryTarget(self.project, self.dataset, table_name)

    @staticmethod
    def _sanitize_date(date):
        return str(date).replace("-", "")

    @staticmethod
    def _normalize_table_name(dataset, table, suffix):
        if table:
            result = table
        elif dataset:
            result = dataset
        else:
            raise ValueError("Either table or dataset must be provided.")

        return "{}_{}".format(result, suffix)


def table_str(bq_target):
    # type: (BigqueryTarget) -> str
    """Given a BigqueryTarget returns a string table reference."""
    t = bq_target.table
    return "{}.{}.{}".format(t.project_id, t.dataset_id, t.table_id)
