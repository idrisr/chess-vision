from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chessviz",
    version="0.0.1",
    author="Idris Raja",
    author_email="idris.raja@gmail.com",
    description="A small utility package",
    install_requires=['fastai']
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idrisr/chess-vision",
    project_urls={
        "Bug Tracker": "https://github.com/idrisr/chess-vision/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)
