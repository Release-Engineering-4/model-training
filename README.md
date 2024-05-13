# RE-project-team-4
Public github repository for the course CS4295 Release Engineering for Machine Learning Applications of Team 4.

# Poetry 

This project uses Poetry for dependency management.

## Usage

To install poetry use: 

```bash
pip install poetry
```
To install project dependencies use:

```bash
poetry install
```

To update dependencies use: 

```bash
poetry update
```

To add extra dependencies to .toml file use:

```bash
poetry add `package-name`
```

To remove existing dependencies from .toml file use:

```bash
poetry remove `package-name`
```

# DVC

This project uses DVC as the version-control system.

## Usage

To push data to the remote storage use: 

```bash
dvc push
```

To pull data from the remote storage use: 

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




