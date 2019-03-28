import luigi
from tfx.components.base.base_component import ComponentOutputs, BaseComponent
from tfx.orchestration import tfx_runner
from tfx.utils.channel import Channel
from tfx.utils.types import TfxType

from spotify_tensorflow.luigi.tfx_adapter import LuigiAdapter, LuigiComponent


class LuigiRunner(tfx_runner.TfxRunner):
    """Tfx runner on Luigi."""

    def __init__(self):
        super(LuigiRunner, self).__init__()

    def run(self, pipeline):
        luigi_components = list()  # type: list[LuigiComponent]
        output_sources = dict()
        # construct luigi component and output_sources dict (channel_name -> producer)
        for component in pipeline.components:  # type: BaseComponent
            # type: dict[str, list[TfxType]]
            input_dict = self._prepare_input_dict(component.input_dict)
            # type: dict[str, list[TfxType]]
            output_dict = self._prepare_output_dict(component.outputs)
            luigi_component = LuigiComponent(
                component_name=component.component_name,
                unique_name=component.unique_name,
                driver=component.driver,
                executor=component.executor,
                input_dict=input_dict,
                output_dict=output_dict,
                exec_properties=component.exec_properties
            )
            luigi_components.append(luigi_component)
            for key in output_dict:
                tfx_type = output_dict[key][0]
                output_sources[tfx_type.type_name] = luigi_component

        luigi_tasks = dict()
        for component in luigi_components:
            input_dict = component.input_dict
            required_components = set()
            for key in input_dict:
                tfx_type = input_dict[key][0]
                required_components.add(output_sources[tfx_type.type_name])
            component.required_components = list(required_components)
            task_name = self._prepare_luigi_task_name(component.component_name,
                                                      component.unique_name)
            task = LuigiAdapter(task_name=task_name)
            task.component = component
            luigi_tasks[task_name] = task

        for component in luigi_components:
            required_tasks = list()
            for required_component in component.required_components:
                required_task_name = self._prepare_luigi_task_name(
                    required_component.component_name, required_component.unique_name)
                required_tasks.append(luigi_tasks[required_task_name])
            task_name = self._prepare_luigi_task_name(component.component_name,
                                                      component.unique_name)
            luigi_tasks[task_name].set_requires(required_tasks)

        tasks = luigi_tasks.values()
        luigi.build(tasks, local_scheduler=True)

    def _prepare_output_dict(self, outputs):  # type: (ComponentOutputs) -> dict[str, list[TfxType]]
        return dict((k, v.get()) for k, v in outputs.get_all().items())

    def _prepare_input_dict(self, input_dict):  # type: (dict[str, Channel]) -> dict[str, list[TfxType]]  # noqa: E501
        return dict((k, v.get()) for k, v in input_dict.items())

    def _prepare_luigi_task_name(self, component_name, unique_name):
        return "{}{}".format(component_name, "" if unique_name is None else "." + unique_name)
