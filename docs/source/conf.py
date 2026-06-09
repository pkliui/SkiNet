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
    'sphinxcontrib.autodoc_pydantic',
]

# myst_parser: enable {autoclass} / {autofunction} directives inside .md files
myst_enable_extensions = ["colon_fence"]

# Auto-generate anchors for h1-h3 headings so cross-file links like
# [text](development.md#lightning-studio) resolve to the heading slug.
myst_heading_anchors = 3

# sphinx.ext.napoleon: use Sphinx-style :param:/:return: (not Google/NumPy style)
napoleon_use_param = True
napoleon_use_rtype = True

# Put types in the signature box (InnerEye style), not scattered in the body
autodoc_typehints = "signature"

# autodoc-pydantic: show fields with their types, defaults, and descriptions
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_members = True
autodoc_pydantic_model_undoc_members = True
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_default = True
# Use only the class docstring — Pydantic generates noisy __init__ boilerplate
autoclass_content = "class"
# Global autodoc defaults for all autoclass/autofunction directives
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
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
    r"https://challenge\.isic-archive\.com/.*",
]


# -- HTML output -------------------------------------------------------------
# html_theme = 'furo'
html_theme = "pydata_sphinx_theme"
