"""
Do not run this in the root directory.
Run build.sh
"""

import setuptools


with open("../README.md", "r") as fp:
    long_description = fp.read()

"""
with open("../requirements.txt", "r") as fp:
    requirements = fp.read().strip().split("\n")
"""
requirements = []

setuptools.setup(
    name="starst3r",
    version="0.0.1",
    author="Patrick Huang",
    author_email="phuang1024@gmail.com",
    description="Mast3r reconstruction and localization with 3DGS view rendering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phuang1024/Starst3r",
    py_modules=["starster", "mast3r", "dust3r", "croco"],
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
