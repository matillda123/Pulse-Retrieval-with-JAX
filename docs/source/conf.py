project = 'PulsedJAX: Pulse-Retrieval-with-JAX'
copyright = '2025, Till Stehling'
author = 'Till Stehling'
release = '2025'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../pulsedjax'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # for Google/NumPy style docstrings
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'myst_nb',
    'sphinx_copybutton'
]

myst_enable_extensions = [
    "dollarmath",  # Enable $...$ and $$...$$ for math
    "amsmath",     # Enable advanced math environments
]


autosummary_generate = True

templates_path = ['_templates']
include_patters = ['**']
exclude_patterns = []

autodoc_member_order = 'bysource'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_theme_options = {
    "collapse_navigation": False,
    "titles_only": True,
}


