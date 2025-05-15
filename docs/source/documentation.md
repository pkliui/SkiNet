# Documenation

This documentation is built with sphinx. To complete the documentation further:


## Make a new conda environment 
- It requires a new conda environment (that is different from the project's environment) as specified in ```skinet/environment_docs.yaml```

```yaml
# this is a file to make an environment for the documentation of the SkiNet project
name: skinet-docs
channels:
  - defaults
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

- Also there is a separate git branch for documentation - ```documentation```
```bash
git checkout -b documentation
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

## Deploy on Github Pages

To publish the documentation on Github Pages, please follow these instructions:

- Make an empty "gh-pages" branch where we will publish the documentation
- Check out at the branch containing the documentation, i.e. "documentation"
```bash
conda activate documentation
```
- Put the following .yaml file under ```.github/workflow```  at the project's root to specify the workflow for building documentation:
e.g. build-docs.yaml file

```yaml
name: Workflow to update documentation index.html each time the code in docs/source changes

on:
  push:
    branches:
      - documentation

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - name: Check out at branch documentation
        uses: actions/checkout@v4

      - name: Set up Miniconda
        uses: conda-incubator/setup-miniconda@v3
        with:
          activate-environment: skinet-docs
          environment-file: environment_docs.yaml
          auto-activate-base: false

      - name: Build the documentation using Sphinx
        shell: bash -l {0}  # -l to ensure a login bash, where the conda environment is correctly set; {0} a template placeholder, replaced at pipeline execution time by the actual script command to execute
        run: |
          sphinx-build -b html docs/source/ docs/build/html
          touch docs/.nojekyll

      - name: Deploy the documentation
        uses: peaceiris/actions-gh-pages@v4
        if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/documentation' }}
        with:
          publish_branch: gh-pages
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/build/html
          force_orphan: true
```

- Under "docs" folder please add an empty .nojekyll file 
- Under the same "docs" folder add an "index.html" file with the following content:
- 
```html
<meta http-equiv="refresh" content="0; url=./build/html/index.html" />
```

- In Settings/Actions/General/Workflow permissions, make sure to have "Read and write permissions" selected.
- Now each time, one pushes to the "documentation" branch, Github Actions will execute this workflow and publish the updated documentation in "gh-pages" branch