"""
Do not run this in the root directory.
Run build.sh

This script will be copied and run from the ``build`` directory,
so all paths are relative to that directory.
"""

import setuptools


def read_reqs(path):
    with open(path, "r") as fp:
        return fp.read().splitlines()


with open("../README.md", "r") as fp:
    long_description = fp.read()

requirements = []
req_paths = (
    "../requirements.txt",
    "../mast3r/requirements.txt",
    "../mast3r/dust3r/requirements.txt",
)
for path in req_paths:
    requirements.extend(read_reqs(path))

setuptools.setup(
    name="starst3r",
    version="0.4.0",
    author="Patrick Huang",
    author_email="phuang1024@gmail.com",
    description="Ultra fast 3D reconstruction and novel view synthesis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phuang1024/Starst3r",
    py_modules=["starster", "mast3r", "dust3r", "croco"],
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
