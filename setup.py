import os
import re
import codecs
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    entry_points={
        "console_scripts": ["blendsql=blendsql.blend_cli:main"],
    },
    name="blendsql",
    version=find_version("blendsql", "__init__.py"),
    url="https://github.com/parkervg/blendsql",
    author="Parker Glenn",
    author_email="parkervg5@gmail.com",
    description="Orchestrate SQLite logic and LLM reasoning within a unified dialect.",
    long_description="BlendSQL is a scalable SQL dialect for problem decomposition and heterogenous question-answering with LLMs. It builds off of the syntax of SQLite to create an intermediate representation for tasks requiring complex reasoning over both structured and unstructured data.",
    license="Apache License 2.0",
    packages=find_packages(exclude=["examples", "research", "img"]),
    install_requires=[
        "openai==0.28.0",
        "guidance==0.0.64",
        "pyparsing==3.1.1",
        "pandas>=2.0.0",
        "bottleneck>=1.3.6",
        "sqlglot",
        "pre-commit",
        "attrs",
        "tqdm",
        "dateparser",
        "colorama",
        "fiscalyear",
        "tabulate",
        "typeguard",
        "azure-identity",
        "nbformat",
    ],
    extras_require={
        "research": [
            "datasets==2.16.1",
            "nltk",
            "wikiextractor",
            "rouge_score",
            "rapidfuzz",
            "records",
            "SQLAlchemy",
            "recognizers-text",
            "recognizers-text-suite",
            "emoji==1.7.0",
            "transformers",
        ],
        "test": [
            "pytest",
        ],
    },
)
