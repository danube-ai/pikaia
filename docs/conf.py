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
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.napoleon",
]

# -- AutoAPI configuration ---------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../pikaia"]
# Exclude "imported-members" to prevent re-exported symbols from being documented
# twice (once at the definition site and once at the re-export site), which would
# otherwise produce "duplicate object description" and "more than one target found"
# warnings throughout the build.
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

# -- Napoleon (Google-style docstring) configuration -------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
