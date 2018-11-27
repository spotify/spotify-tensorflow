FROM gcr.io/cloud-dataflow/v1beta3/python-fnapi:2.8.0

# TODO(brianm): https://github.com/tensorflow/data-validation/issues/41
#   Installing without version pinning might mean that the
#   worker and driver have version mismatches. We should prefer to install
#   the entire reqs.txt or similar.
RUN pip install tensorflow-data-validation
