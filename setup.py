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
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(exclude=["examples", "research", "img"]),
    install_requires=[
        "guidance>=0.1.0",
        "pyparsing==3.1.1",
        "pandas>=2.0.0",
        "bottleneck>=1.3.6",
        "python-dotenv==1.0.1",
        "sqlglot==18.13.0",
        "platformdirs",
        "pre-commit",
        "attrs",
        "tqdm",
        "dateparser",
        "colorama",
        "fiscalyear",
        "tabulate",
        "typeguard",
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
        ],
        "test": ["pytest", "huggingface_hub"],
        "docs": [
            "mkdocs-material",
            "mkdocstrings",
            "mkdocs-section-index",
            "mkdocstrings-python",
            "mkdocs-jupyter",
        ],
    },
)
