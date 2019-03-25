from tfx.orchestration import tfx_runner

from spotify_tensorflow.luigi.luigi_adapter import LuigiAdapter


class LuigiRunner(tfx_runner.TfxRunner):
    """Tfx runner on Airflow."""

    def __init__(self):
        super(LuigiRunner, self).__init__()

    def run(self, pipeline):
        luigi_tasks = []
        # TODO: find a better way to determine the luigi job trigger
        # luigi assumes the last component in the pipeline is the trigger
        trigger = pipeline.components[-1]
        return LuigiAdapter(trigger)

