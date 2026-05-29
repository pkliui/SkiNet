import sys
import os

# Make the SkiNet package importable by autodoc
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'SkiNet'
copyright = '2024, Pavel Kliuiev'
author = 'Pavel Kliuiev'
release = '1.0.0'


# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx_togglebutton',
    'sphinx.ext.autodoc',       # pulls docstrings from source into API pages
    'sphinx.ext.napoleon',      # understands :param:/:return: Sphinx style
    'sphinx.ext.viewcode',      # adds [source] links next to every documented symbol
    'sphinx.ext.intersphinx',   # lets {py:class}`torch.Tensor` resolve to PyTorch docs
]

# myst_parser: enable {autoclass} / {autofunction} directives inside .md files
myst_enable_extensions = ["colon_fence"]

# sphinx.ext.napoleon: use Sphinx-style :param:/:return: (not Google/NumPy style)
napoleon_use_param = True
napoleon_use_rtype = True

# Put types in the signature box (InnerEye style), not scattered in the body
autodoc_typehints = "signature"
# Use only the class docstring — Pydantic generates noisy __init__ boilerplate
autoclass_content = "class"
# Global autodoc defaults for all autoclass/autofunction directives
autodoc_default_options = {
    "members": False,
    "show-inheritance": True,
}

# intersphinx: resolve cross-refs to upstream libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'
templates_path = ['_templates']
exclude_patterns: list[str] = []

# pytorch.org moved the randomness page; anchor no longer resolves
linkcheck_ignore = [
    r"https://docs\.pytorch\.org/docs/stable/notes/randomness\.html.*",
]


# -- HTML output -------------------------------------------------------------
html_theme = 'furo'
