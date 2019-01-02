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

# flake8: noqa

from __future__ import absolute_import, division, print_function

import json
import logging
import subprocess

from luigi import LocalTarget
from mock import patch
from nose.tools import raises
import responses
from spotify_tensorflow.luigi.utils import get_uri, is_gcs_path, _fetch_file, fetch_tfdv_whl, \
    run_with_logging, to_snake_case
from tensorflow.python.platform import test


class LuigiUtilsTest(test.TestCase):

    @staticmethod
    def test_to_snake_case():
        assert "a.b.c" == to_snake_case("A.B.C")
        assert "a.bc" == to_snake_case("A.BC")
        assert "a.b_c" == to_snake_case("A.B_C")
        assert "base_class" == to_snake_case("BaseClass")
        assert "base_class_y" == to_snake_case("BaseClassY")

        # Is this really desired behavior?
        assert "a._base_class" == to_snake_case("A.BaseClass")

    @staticmethod
    def test_is_gcs_path():
        assert is_gcs_path("gs://something")
        assert is_gcs_path("gs://something/else")
        assert is_gcs_path("GS://something/else")
        assert is_gcs_path("gS://something")
        assert is_gcs_path("gS://something ")
        assert is_gcs_path(" gS://something ")
        assert not is_gcs_path("file://something")
        assert not is_gcs_path("something")

    @staticmethod
    def test_get_uri():

        class TargetWithURI(LocalTarget):
            def uri(self):
                return 'i://have/a/uri'

        class TargetWithPath(LocalTarget):
            pass

        class NotATarget:
            pass

        assert get_uri(TargetWithURI('fake/path')) == 'i://have/a/uri'
        assert get_uri(TargetWithPath('a/path')) == 'a/path'

        try:
            get_uri(NotATarget())
            assert False
        except ValueError as e:
            assert "Unknown input target type" in e.message

    @staticmethod
    def test_run_with_logging():
        logger = logging.getLogger("luigi-interface")
        actual = run_with_logging(["python", "-c", "print(1)"], logger)
        assert actual == 0

        try:
            run_with_logging(["python", "-c", "import sys; sys.exit(1)"], logger)
            assert False
        except subprocess.CalledProcessError:
            assert True

    @staticmethod
    @responses.activate
    def test_fetch_file_when_200():

        test_content = {"test": "content"}
        responses.add(responses.GET, "http://this-is-a-test.com/file.txt",
                      json=test_content, status=200)

        output_path = _fetch_file("http://this-is-a-test.com/file.txt")

        assert test_content == json.load(open(output_path, 'r'))


    @staticmethod
    @raises(Exception)
    @responses.activate
    def test_fetch_file_when_404():
        test_content = {"test": "missing-content"}
        responses.add(responses.GET, "http://this-is-a-test.com/file.txt",
                      json=test_content, status=404)

        _fetch_file("http://this-is-a-test.com/file.txt")

    @staticmethod
    @patch("tensorflow_data_validation.__version__", "0.11.0")
    @responses.activate
    def test_fetch_tfdv_whl():
        test_wheel = {"I": "is-a-wheel"}
        responses.add(responses.GET,
                      "https://files.pythonhosted.org/packages/f0/f1"
                      "/c3441933b8a5fe0737dab7850804c7cec3f5fe7b2cc609dd1ce5987df768"
                      "/tensorflow_data_validation-0.11.0-cp27-cp27mu-manylinux1_x86_64.whl",
                      status=200, json=test_wheel)

        responses.add(responses.GET, "https://pypi.org/simple/tensorflow_data_validation",
                      status=200, body="""
<!DOCTYPE html>
<html>
<head>
<title>Links for tensorflow-data-validation</title>
</head>
<body>
<h1>Links for tensorflow-data-validation</h1>
    <a href="https://files.pythonhosted.org/packages/8f/cc/da87af98d01014f669c6f9dd38b5c04fdd8d2a66a948479db3a4a1a958d9/tensorflow_data_validation-0.9.0-cp27-cp27m-macosx_10_11_x86_64.whl#sha256=95b01888c04a49b834ee6b2fbdae0d2406346eb4ec579d3b71e9c70be4635bd3" data-requires-python="&gt;=2.7,&lt;3">tensorflow_data_validation-0.9.0-cp27-cp27m-macosx_10_11_x86_64.whl</a><br/>
    <a href="https://files.pythonhosted.org/packages/93/d7/da5c07da538b7d13f115ee94bddbb91fb4da01226a0cd168dcea68bac596/tensorflow_data_validation-0.9.0-cp27-cp27m-macosx_10_12_x86_64.whl#sha256=5f90501d99d72f322a3bad019ce017d9c754a3d986249cc7f5867bf0d98be441" data-requires-python="&gt;=2.7,&lt;3">tensorflow_data_validation-0.9.0-cp27-cp27m-macosx_10_12_x86_64.whl</a><br/>
    <a href="https://files.pythonhosted.org/packages/95/b4/9dff633f413baad469ee7039b4477150a5e5ed32a3210966cbaaf0bb6fed/tensorflow_data_validation-0.9.0-cp27-cp27mu-manylinux1_x86_64.whl#sha256=2ba5fabab3f2146778da322688ae918091b9b1cd0fc50bfa22ffe65be5430396" data-requires-python="&gt;=2.7,&lt;3">tensorflow_data_validation-0.9.0-cp27-cp27mu-manylinux1_x86_64.whl</a><br/>
    <a href="https://files.pythonhosted.org/packages/db/44/3231030547d44e8fe7f4325db3b5077d4710678e4ea116cade0940b204d7/tensorflow_data_validation-0.11.0-cp27-cp27m-macosx_10_11_x86_64.whl#sha256=35659101b0751903d1443b5a4dd74b0ee6b73da871b20fa21288f7713c9b14e6" data-requires-python="&gt;=2.7,&lt;3">tensorflow_data_validation-0.11.0-cp27-cp27m-macosx_10_11_x86_64.whl</a><br/>
    <a href="https://files.pythonhosted.org/packages/65/6a/b6b199edbf16ddca265b00ca4ebebace088dda2463ed24b94403d19cabb9/tensorflow_data_validation-0.11.0-cp27-cp27m-macosx_10_12_x86_64.whl#sha256=f0ab3a6927034922e8efdcb66251d74d2c5224803d044833d01c707a4c6c6f89" data-requires-python="&gt;=2.7,&lt;3">tensorflow_data_validation-0.11.0-cp27-cp27m-macosx_10_12_x86_64.whl</a><br/>
    <a href="https://files.pythonhosted.org/packages/f0/f1/c3441933b8a5fe0737dab7850804c7cec3f5fe7b2cc609dd1ce5987df768/tensorflow_data_validation-0.11.0-cp27-cp27mu-manylinux1_x86_64.whl#sha256=055593d7bdcac0bf84408d20189da7951f840cddde43aaa3c4c4da9038783460" data-requires-python="&gt;=2.7,&lt;3">tensorflow_data_validation-0.11.0-cp27-cp27mu-manylinux1_x86_64.whl</a><br/>
</body>
</html>
<!--SERIAL 4483951-->
""")

        # Test with explicit version given
        local_whl = fetch_tfdv_whl("0.11.0")
        assert test_wheel == json.load(open(local_whl, "r"))
        assert local_whl.endswith(
            "tensorflow_data_validation-0.11.0-cp27-cp27mu-manylinux1_x86_64.whl")

        # Test without providing version
        local_whl = fetch_tfdv_whl()
        assert test_wheel == json.load(open(local_whl, "r"))
        assert local_whl.endswith(
            "tensorflow_data_validation-0.11.0-cp27-cp27mu-manylinux1_x86_64.whl")

    @staticmethod
    @responses.activate
    def test_fetch_tfdv_whl_bad_response():
        test_wheel = {"I": "is-a-wheel"}
        responses.add(responses.GET,
                      "https://files.pythonhosted.org/packages/f0/f1"
                      "/c3441933b8a5fe0737dab7850804c7cec3f5fe7b2cc609dd1ce5987df768"
                      "/tensorflow_data_validation-0.11.0-cp27-cp27mu-manylinux1_x86_64.whl",
                      status=200, json=test_wheel)

        responses.add(responses.GET, "https://pypi.org/simple/tensorflow_data_validation",
                      status=200, body="""
<!DOCTYPE html>
<html>
... garbled response ...
unparseable
""")

        try:
            fetch_tfdv_whl("0.11.0")
            assert False
        except Exception as e:
            assert "Problem fetching package. Couldn't parse listing" in e.message
