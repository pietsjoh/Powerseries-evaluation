"""Creates docs/source/conf.py and docs/source/index.rst.
These are the configuration files for the documentation.
Executed during build process.
"""
if __name__ == "__main__":
    from pathlib import Path
    headDirPath = Path(__file__).parents[2]
    docsSourcePath = (headDirPath / "docs" / "source").resolve()

    ## format of conf.py file
    confpyFile = """# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
from pathlib import Path
headDirPath = Path(__file__).parents[2]
srcDirPath = (headDirPath / "pysrc").resolve()
sys.path.append(str(srcDirPath))


# -- Project information -----------------------------------------------------

project = 'Peak data evaluation'
copyright = '2021, Johannes Pietsch'
author = 'Johannes Pietsch'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# favicon (image in the tab or bookmarks)
html_favicon = str((headDirPath / "res" / 'jp_logo_schnitt.jpg').resolve())"""

    ## Writing / creating conf.py file
    confpyPath = str((docsSourcePath / "conf.py").resolve())
    with open(confpyPath, "w") as f:
        f.write(confpyFile)

    ## format of index.rst file
    indexrstFile = """.. Peak data evaluation documentation master file, created by
   sphinx-quickstart on Wed May 26 18:39:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Peak data evaluation's documentation!
================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   data_tools
   FSR_analysis
   peak_fit
   powerseries
   setup
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`"""

    ## write / create index.rst file
    indexrstPath = str((docsSourcePath / "index.rst").resolve())
    with open(indexrstPath, "w") as f:
        f.write(indexrstFile)