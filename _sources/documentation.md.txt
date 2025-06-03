# Documenation

At present the documentation is built with sphinx in the dev branch and its remote version (GitHub Pages site https://pkliui.github.io/SkiNet/) is updated upon a push to the dev branch as specified in the workflow (see below).

To complete the documentation further:

## Building documentation locally

### Make a new conda environment 
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
conda activate skinet-docs
```


### Configure documentation

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

- File ```docs/source/index.rst``` serves as a welcome page and contains the root of the “table of contents tree”. 

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


### Build documentation
- Given you are in the project folder containing the ```docs``` folder, run:

```bash
sphinx-build -M html docs/source/ docs/build/
```

- Now the documentation is ready under ```SkiNet/docs/build/html/index.html```. 

- Check that all links are working with

```bash
sphinx-build docs/source -W -b linkcheck -d docs/source docs/build/html
```

## Deploy on Github Pages

To publish the documentation on Github Pages, please follow these instructions:

-  Make an empty "gh-pages" branch where the documentation will be published as a GitHub Pages site
-  Check out at the git branch containing the documentation locally, i.e. dev at the time of writing
  
```bash
git checkout dev
```
- Put the following .yaml file under ```.github/workflow```  to specify the workflow for building documentation:
e.g. build-docs.yaml file

```yaml
name: Workflow to update documentation upon push to dev branch

on:
  push:
    branches:
      - dev

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - name: Check out at the current branch
        uses: actions/checkout@v4

      - name: Set up conda environment "skinet-docs" to build documentation
        uses: conda-incubator/setup-miniconda@v3
        with:
          activate-environment: skinet-docs
          environment-file: environment_docs.yaml
          auto-activate-base: false

      - name: Build the documentation using Sphinx
        shell: bash -l {0}  # -l to ensure a login bash, where the conda environment is correctly set; {0} a template placeholder, replaced at pipeline execution time by the actual script command to execute
        run: | # build sphinx documentation directly into /docs to have index.html in /docs; this is expected so when the GitHub Pages site is being built from branch "gh-pages", folder "/root" (see Settings/Pages/Build and deployment)
          sphinx-build -b html docs/source/ docs/ 
          touch docs/.nojekyll 

      - name: Deploy the documentation to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/dev' }}
        with: # the branch specified in "publish_branch" should be the same as in Settings/Pages, e.g. "gh-pages" and folder "/root"
          publish_branch: gh-pages
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/
          force_orphan: true

```

- Under "docs" folder please add an empty ".nojekyll" file. 
      - The ".nojekyll" file is used to disable Jekyll processing on GitHub Pages and ensure sphinx-generated documentation is correctly served

- Push the changes to remote

- In Settings/Actions/General/Workflow permissions, make sure to have "Read and write permissions" selected.

- Under Settings/General/Pages, make sure to select  "Deploy from branch", Branch "gh-pages", "/root" folder (NOT "dev" branch!)
  
- Now each time, one pushes to the "dev" branch, Github Actions will execute this workflow and publish the updated documentation in "gh-pages" branch