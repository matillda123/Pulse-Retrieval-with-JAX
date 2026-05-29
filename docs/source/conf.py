project = 'PulsedJAX: Pulse-Retrieval-with-JAX'
copyright = '2025, Till-Jakob Stehling'
author = 'Till-Jakob Stehling'
release = '2025'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../pulsedjax'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx.ext.linkcode',
    'sphinx.ext.graphviz',
    'sphinxcontrib.bibtex'
]

bibtex_bibfiles = ['pulsedjax_literature.bib']
bibtex_default_style = "unsrt"

myst_enable_extensions = [
    "dollarmath",  # Enable $...$ and $$...$$ for math
    "amsmath",     # Enable advanced math environments
]
nb_execution_mode = "off"

autosummary_generate = True
autosummary_imported_members = True

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




import inspect
import pulsedjax

def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    if not modname:
        return None

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    if not fn:
        return None

    # Make path relative to package root
    fn = os.path.relpath(fn, start=os.path.dirname(pulsedjax.__file__))

    end_line = lineno + len(source) - 1

    return (
        "https://github.com/matillda123/Pulse-Retrieval-with-JAX/tree/main/"
        f"pulsedjax/{fn}#L{lineno}-L{end_line}"
    )