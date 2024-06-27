# RE-project-team-4

![CI](https://github.com/Release-Engineering-4/model-training/actions/workflows/pipeline.yml/badge.svg)

Public github repository for the course CS4295 Release Engineering for Machine Learning Applications of Team 4.

# Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management.
To ensure a consistent environment setup, follow the steps below.

## Usage

To install poetry use:

```bash
pip install poetry
```

To install project dependencies use:

```bash
poetry install
```

**OPTIONAL STEPS**

To add extra dependencies to `pyproject.toml` file use:

```bash
poetry add `package-name`
```

To remove existing dependencies from `pyproject.toml` file use:

```bash
poetry remove `package-name`
```

To update dependencies use:

```bash
poetry update
```

# DVC

This project uses [DVC](https://dvc.org/) as the version-control system.
To reproduce our ML pipeline experiment, follow the steps below.

## Usage

To ensure you are working with the latest file versions (data/model/tokenizer) from Google Drive, use:

```bash
dvc pull
```

To run the pipeline stages use:

```bash
dvc repro
```

To show all the metrics use:

```bash
dvc metrics show
```

**OPTIONAL STEP**

To upload files to remote storage (in case of significant changes), use:

```bash
dvc push
```

## Testing

To run the tests for the pre-processing library use:

```bash
pytest
```

To run the tests with coverage for the pre-processing library use:

```bash
coverage run -m pytest -i
```

To generate the coverage report use:

```bash
coverage report -m -i
```

To generate the html of the coverage report use:

```bash
coverage html -i
```

# Cookiecutter

The project template was created using [Cookiecutter](https://www.cookiecutter.io/).

# Pylint & Flake8

To analyze code for errors, enforce coding standards, and look for code smells, we employ [Pylint](https://pylint.readthedocs.io/en/stable/) and [Flake8](https://flake8.pycqa.org/en/latest/).

# Support

If you encounter any problems or bugs with `model-training`, feel free to open an issue on the project repository.
