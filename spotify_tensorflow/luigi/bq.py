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

import luigi
from luigi.contrib.bigquery import BigqueryTarget

from external_daily_snapshot import ExternalDailySnapshot


class BigQueryDailySnapshot(ExternalDailySnapshot):
    project = luigi.Parameter()
    dataset = luigi.Parameter()
    table = luigi.Parameter(default=None)

    def output(self):
        date_replace = str(self.date).replace("-", "")
        table = self.table if self.table else self.dataset
        table_name = "%s_%s" % (table, date_replace)
        return BigqueryTarget(self.project, self.dataset, table_name)


def table_str(bq_target):
    t = bq_target.table
    return "%s.%s.%s" % (t.project_id, t.dataset_id, t.table_id)
