import logging
import os
import tempfile

import luigi
import textwrap
from spotify_hades import HadesTarget

logger = logging.getLogger("luigi-interface")


class LuigiComponent(object):
    def __init__(self, component_name, unique_name, driver, executor,
                 input_dict, output_dict, exec_properties):
        self.component_name = component_name
        self.unique_name = unique_name
        self.driver = driver
        self.executor = executor
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.exec_properties = exec_properties
        self._required_components = list()

    @property
    def required_components(self):
        return self._required_components

    @required_components.setter
    def required_components(self, components):
        self._required_components = components


class LuigiAdapter(luigi.Task):
    task_name = luigi.Parameter()
    output_dir = "gs://ml-sketchbook-keshi/tfx-pipeline"
    service_account_email = "keshi-paved-road@ml-sketchbook.iam.gserviceaccount.com"
    _required_tasks = list()
    _component = None  # type: LuigiComponent
    _targets = list()  # type: list[TfxTypeTarget]

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component):
        self._component = component
        output_dict = component.output_dict
        for key in output_dict:
            self._targets.extend([TfxTypeTarget(component_name=self.component.component_name,
                                                tfx_type=tfx_type,
                                                uri_prefix=self.output_dir)
                                  for tfx_type in output_dict[key]])

    def set_requires(self, required_tasks):  # type: (list[luigi.Task]) -> None
        self._required_tasks = required_tasks

    def requires(self):
        return self._required_tasks

    def run(self):
        input_dict = self.component.input_dict
        output_dict = self.component.output_dict
        exec_properties = self.component.exec_properties
        beam_pipeline_args = ["--runner=DataflowRunner",
                              "--experiments=shuffle_mode=auto",
                              "--project=ml-sketchbook",
                              "--max_num_workers=25",
                              "--worker_machine_type=n1-standard-32",
                              "--temp_location=" + os.path.join(self.output_dir, "tmp"),
                              "--service_account_email=" + self.service_account_email,
                              "--setup_file=" + create_setup_file()]
        executor = self.component.executor(beam_pipeline_args=beam_pipeline_args)
        executor.Do(input_dict=input_dict, output_dict=output_dict, exec_properties=exec_properties)
        for target in self._targets:
            target.publish()

    def output(self):
        return self._targets


class TfxTypeTarget(luigi.Target):
    def __init__(self, component_name, tfx_type, uri_prefix):
        self.component_name = component_name
        self.tfx_type = tfx_type
        endpoint = self._prepare_endpoint_name(component_name,
                                               tfx_type.type_name,
                                               tfx_type.split)
        partition = str(self.tfx_type.span)
        self.hades = HadesTarget(endpoint=endpoint, partition=partition,
                                 uri_prefix=uri_prefix)  # noqa: E501
        self._uri = self.hades.generate_uri()
        self.tfx_type.uri = self._uri

    def path(self):
        return self.uri()

    def uri(self):
        return self._uri

    def exists(self):
        return self.hades.exists()

    def publish(self):
        logger.info("----publish endpoint----")
        logger.info(self.hades.endpoint)
        logger.info("------------------------")
        self.hades.publish(self._uri)

    def _prepare_endpoint_name(self, component_name, type_name, split):
        return "{}.{}{}".format(component_name, type_name, "." + split if split else "")


def create_setup_file():
    contents_for_setup_file = """
    import setuptools
    
    if __name__ == "__main__":
        setuptools.setup(
            name="spotify_tensorflow_dataflow",
            packages=setuptools.find_packages(),
            install_requires=[
                "tfx"
        ])
    """  # noqa: W293
    setup_file_path = os.path.join(tempfile.mkdtemp(), "setup.py")
    with open(setup_file_path, "w") as f:
        f.writelines(textwrap.dedent(contents_for_setup_file))
    return setup_file_path
