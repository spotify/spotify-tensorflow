#!/usr/bin/env python
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
import codecs
import os
import re

from setuptools import find_packages
from setuptools import setup


HERE = os.path.abspath(os.path.dirname(__file__))


#####
# Helper functions
#####
def read(*filenames, **kwargs):
    """
    Build an absolute path from ``*filenames``, and  return contents of
    resulting file.  Defaults to UTF-8 encoding.
    """
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for fl in filenames:
        with codecs.open(os.path.join(HERE, fl), "rb", encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


def find_meta(meta):
    """Extract __*meta*__ from META_FILE."""
    re_str = r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta)
    meta_match = re.search(re_str, META_FILE, re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


def read_requirements(filename):
    reqs_txt = read(filename)
    parsed = reqs_txt.split("\n")
    parsed = [r.split("==")[0] for r in parsed]
    return [r for r in parsed if len(r) > 0]


#####
# Project-specific constants
#####
NAME = "spotify-tensorflow"
KEYWORDS = ["tensorflow", "utils"]
PACKAGE_NAME = "spotify_tensorflow"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join(PACKAGE_NAME, "__init__.py")
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2.7"
]
META_FILE = read(META_PATH)


setup(
    name=NAME,
    version=find_meta("version"),
    description=find_meta("description"),
    url=find_meta("url"),
    keywords=KEYWORDS,
    license=find_meta("license"),
    packages=PACKAGES,
    python_requires=">=2.7,<3",
    include_package_data=True,
    classifiers=CLASSIFIERS,
    zip_safe=False,
    install_requires=read_requirements("requirements.txt"),
    test_suite="tests",
    entry_points={
        "console_scripts": [
            "tfr-read = spotify_tensorflow.scripts.tfr_read:main"
        ]
    },
)
