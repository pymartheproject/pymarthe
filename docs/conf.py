# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.join('..','pymarthe'))
sys.path.append(os.path.join(".."))
from pymarthe import __version__


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyMarthe'
copyright = '2024, Pryet, A., Matran P., Aissat R., Buscarlet E., Manceau, JC, Vergnes, J.P.'
author = 'Pryet, A., Matran P., Aissat R., Buscarlet E., Manceau, JC, Vergnes, J.P.'
release = str(__version__)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon','sphinx.ext.githubpages','sphinx.ext.autosummary']

#autosummary_generate = True
extensions.append('autoapi.extension')

autoapi_type = 'python'
autoapi_dirs = [os.path.join('..','pymarthe')]
autoapi_ignore = ['*version*']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
