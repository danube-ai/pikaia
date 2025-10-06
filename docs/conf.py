# Configuration file for the Sphinx documentation builder.

# -- Add project root to sys.path for AutoAPI import resolution --
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pikaia"
copyright = "2025, danube.ai"
author = "danube.ai"
release = "0.04"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "myst_parser",
]
autoapi_type = "python"
autoapi_dirs = ["../pikaia"]  # path to your package
html_theme = "sphinx_rtd_theme"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
