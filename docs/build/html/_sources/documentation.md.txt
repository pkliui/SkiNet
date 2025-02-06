# Documenation

This documenation is built with sphinx. To complete the documentation further:


## Make a new conda environment 
- It requires a new conda environment (that is different from the project's environment) as specified in ```skinet/environment_docs.yaml```

```yaml
# this is a file to make an environment for the documentation of the SkiNet project
name: skinet-docs
channels:
  - conda-forge
dependencies:
  - sphinx
  - furo
  - myst-parser
  - sphinx-togglebutton
```

```bash
conda env create -f environment_docs.yaml
```
```bash
conda activate skinet-docs
```

- Also there is a separate git branch for documentation - ```docs```
```bash
git checkout -b docs
```


## Configure documentation

- Following the instructions on https://www.sphinx-doc.org/en/master/tutorial/getting-started.html, go to the project's root directory and run 
```bash
sphinx-quickstart docs
```

- Once prompted, select "separate source and build directories". You will be presented with this filestructure:

```
SkiNet
└── docs
|    ├── build
|    ├── make.bat
|    ├── Makefile
|    └── source
|        ├── conf.py
|        ├── index.rst
|        ├── _static
|        └── _templates
└── SkiNet
|    ├── project's modules
```

- ```docs/source/index.rst``` serves as a welcome page and contains the root of the “table of contents tree”. 

- Whenever you add a new file (page) to your project, add it to the Contents section under the "toctree", e.g.:
```

Contents
--------

.. toctree::

   documentation
   dataset

```



- Customize the documentation using ```conf.py``` file. Example:

```python
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SkiNet'
copyright = '2024, Pavel Kliuiev'
author = 'Pavel Kliuiev'
release = '1.0.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# simply add the extension to your list of extensions
# myst_parser is needed to integrate markdown with sphinx
# sphinx_togglebutton is to add collapsible content to your documentation
extensions = ['myst_parser', "sphinx_togglebutton"]

# specify valid source files extensions to use for docs - both markdown and RST
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import furo
html_theme = 'furo'

html_static_path = ['_static']
```


## Build documentation
- Given you are in the project folder containing the ```docs``` folder, run the build:

```bash
sphinx-build -M html docs/source/ docs/build/
```

Now it is ready under ```SkiNet/docs/build/html/index.html```. 

- Check that all links are working with

```bash
sphinx-build docs/source -W -b linkcheck -d docs/source docs/build/html
```