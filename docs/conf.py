# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import re
import datetime

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../'))
path = os.path.abspath(os.path.dirname(__file__))

# -- Project information -----------------------------------------------------
project = 'Loupe'
author = 'Andy Kee'
copyright = f'{datetime.datetime.now().year} Andy Kee'

with open(os.path.normpath(os.path.join(path, '..', 'loupe', '__init__.py'))) as f:
    version = release = re.search("__version__ = '(.*?)'", f.read()).group(1)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'show_prev_next': False,
    'github_url': 'https://github.com/andykee/loupe',
    'google_analytics_id': '267751812'
}

html_logo = '_static/img/loupe.png'

html_additional_pages = {
    'index': 'indexcontent.html'
}

html_show_sphinx = False

html_show_sourcelink = False

html_scaled_image_link = False

html_js_files = ['js/copybutton.js']

html_css_files = ['css/loupe.css', 'css/syntax-highlighting.css']

pygments_style = 'default'

# if true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

autodoc_default_options = {
    'member-order': 'alphabetical',
    'exclude-members': '__init__, __weakref__, __dict__, __module__',
    'undoc-members': False
}

autosummary_generate = True
