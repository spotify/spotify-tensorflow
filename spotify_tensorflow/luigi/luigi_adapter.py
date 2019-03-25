import luigi

from tfx.components.base.base_component import BaseComponent


class LuigiAdapter(luigi.Task):
    def __init__(self, component):  # type: (BaseComponent) -> None
        self.component = component

    def requires(self):
        pass

    def run(self):
        self.component.executor.Do()

    def output(self):
        pass

