# Contributing

---

## Environment setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pkliui/SkiNet.git
   cd SkiNet
   ```

2. Create the conda environment (requires micromamba or conda):
   ```bash
   micromamba env create -f environment.yaml
   micromamba activate skinet
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Obtain ISIC 2017 data and generate metadata (see [data.md](docs/source/data.md)).

> **Docker-based development:** SkiNet development happens inside a Docker container
> (Ubuntu 22.04 + micromamba, built from the repo `Dockerfile`). For building/running the
> CPU and GPU images, Lightning Studio and Kaggle startup scripts, and the debugging
> cheatsheet, see [development.md](docs/source/development.md).

---

## Branching convention

Branches follow a `type/short-description` pattern:

| Prefix | Use for |
|---|---|
| `feat/` | New features or model variants |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `chore/` | Tooling, CI, environment, refactoring |
| `test/` | New or improved tests |

Base new branches off `dev`. Merge to `dev` via PR; `main` receives stable releases.

---

## PR checklist

Before opening a PR:

_The first four items are enforced automatically by the pre-commit hooks
(see [development.md](docs/source/development.md#linting-and-type-checking)). The last
two are manual review items._

- [ ] All new code is covered by tests in `Tests/`
- [ ] `python -m pytest Tests/ -v` passes locally
- [ ] `flake8 SkiNet/ Tests/` reports no errors
- [ ] `python check_types.py` (mypy via the repo wrapper, using `mypy.ini`) reports no errors (or existing suppressions are documented)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`) — note the `run-pytest-tests` hook runs the full `Tests/` suite on commit
- [ ] _(manual)_ YAML config keys are documented in `docs/source/config.md` if added
- [ ] _(manual)_ Public classes/functions have docstrings

---

## Running tests

```bash
# Full suite
python -m pytest Tests/

# Specific module
python -m pytest Tests/ML/configs/ -v

# With coverage
python -m pytest Tests/ --cov=SkiNet --cov-report=term-missing
```

Tests are organized to mirror the source tree under `SkiNet/`. Shared fixtures live in
`conftest.py` at the repo root.

Azure integration tests (`Tests/Azure/`) require valid Azure credentials and are skipped
in standard CI.

---

## Commit message style

Based on the project's existing git history:

```
TYPE short imperative summary (under 72 chars)

Optional body explaining why, not what.
```

Common prefixes used in this project (from git history): `FEAT`, `FIX`, `DOCS`, `CHORE`, `TESTS`, `REFACTOR`.

---

## Adding a new dataset

1. Create a config class under `SkiNet/ML/configs/data_configs/` that subclasses `BaseDataConfig`.
   Set `METADATA_CSV_NAME`, `REQUIRED_COLUMNS`, and `DATASET_KEY`.
2. Create a CSV builder under `SkiNet/ML/datasets/preprocessing/` that subclasses
   `BaseCSVBuilder`.
3. Register the new kind in `metadata_csv_factory.py`.
4. Add the dataset key to `DatasetKey` in `experiment_keys.py`.
5. Register the config in `experiment_config.py` `DataConfig` discriminated union.
6. Add the Azure path mapping to `azure_settings.yaml` under `PATH_ON_DATASTORE`.
7. Add tests in `Tests/ML/datasets/` and `Tests/ML/configs/`.

---

## Adding a new model

1. Create a config class under `SkiNet/ML/configs/model_configs/` that subclasses `BaseModelConfig`.
   Set `kind` as a `Literal` string.
2. Implement the model in `SkiNet/ML/model/`.
3. Register the config in `experiment_config.py` `ModelConfig` discriminated union.
4. Register the model key in `ModelKey` in `experiment_keys.py`.
5. Add tests in `Tests/ML/model/` (architecture in `architecture/`, building blocks in `blocks/`).
