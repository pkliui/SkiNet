# Documentation

At present the documentation is built with sphinx in the dev branch and its remote version (GitHub Pages site https://pkliui.github.io/SkiNet/) is updated upon a push to the dev branch as specified in the workflow (see below).

To complete the documentation further:

## Building documentation locally (for debugging or preview)

### Quick start: install dependencies with pip on Docker host

```bash
pip install -r requirements_docs.txt
```

This installs `sphinx`, `furo`, `myst-parser`, `sphinx-togglebutton`, `autodoc-pydantic`, `nbsphinx`,
and `ipykernel` (the contents of `requirements_docs.txt`; `nbsphinx`/`ipykernel` render the analysis
notebooks). The CI build uses the equivalent `skinet-docs` conda environment in `environment_docs.yaml`.

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

- Customize the documentation using ```conf.py``` file. The example below is the original
  scaffold; the **current** `docs/source/conf.py` is the source of truth and additionally enables
  `nbsphinx` (renders the analysis notebooks; `nbsphinx_execute = "never"`), `sphinx.ext.autodoc`,
  `sphinx.ext.napoleon`, `sphinx.ext.viewcode`, `sphinx.ext.intersphinx`,
  `sphinxcontrib.autodoc_pydantic`, and `myst_heading_anchors = 3` for the API reference and
  cross-file links. Edit that file directly rather than copying this snippet.

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

or

```bash
python -m sphinx -M html docs/source/ docs/build/
```

- Now the documentation is ready under ```SkiNet/docs/build/html/index.html```.

- Check that all links are working with

```bash
sphinx-build docs/source -W -b linkcheck -d docs/source docs/build/html
```

### Preview in Lightning Studio

- Docs can be previewed on e.g. localhost:8000, by running the following command and remember to enable port mapping from local to the studio
```bash
python -m http.server 8000 --directory docs/build/html
ssh -N -L 8000:localhost:8000 <studio-id>@ssh.lightning.ai
```


## Deploy on Github Pages

To publish the documentation on Github Pages, please follow these instructions:

-  Make an empty "gh-pages" branch where the documentation will be published as a GitHub Pages site
-  Check out at the git branch containing the documentation locally, i.e. dev at the time of writing

```bash
git checkout dev
```

- The deploy workflow lives at [`.github/workflows/build-docs.yaml`](https://github.com/pkliui/SkiNet/blob/dev/.github/workflows/build-docs.yaml) (the source of truth). On every push to `dev` it:
  1. checks out the repo (`actions/checkout@v4`);
  2. creates the `skinet-docs` conda env from `environment_docs.yaml` via `conda-incubator/setup-miniconda@v3` (`activate-environment: skinet-docs`, `auto-activate-base: false`);
  3. **validates links**: `sphinx-build docs/source -b linkcheck -d docs/source docs/build/html`;
  4. **builds HTML into `docs/`** (so `index.html` sits at the publish root): `sphinx-build -b html docs/source/ docs/` then `touch docs/.nojekyll`;
  5. **deploys** to the `gh-pages` branch via `peaceiris/actions-gh-pages@v4` (`publish_dir: docs/`, `publish_branch: gh-pages`, `force_orphan: true`).

- The `docs/.nojekyll` file (created by the workflow) disables Jekyll so the Sphinx `_static`/`_sources` directories are served.

- Push the changes to remote

- In Settings/Actions/General/Workflow permissions, make sure to have "Read and write permissions" selected.

- Under Settings/General/Pages, make sure to select  "Deploy from branch", Branch "gh-pages", "/root" folder (NOT "dev" branch!)

- Now each time, one pushes to the "dev" branch, Github Actions will execute this workflow and publish the updated documentation in "gh-pages" branch
