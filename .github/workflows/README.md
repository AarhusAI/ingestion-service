# GitHub Actions workflows (Python)

These workflows are installed by `itkdev-docker-compose template:install python` and overwritten by
`itkdev-docker-compose template:update`. Do not edit them in the project; make a pull request on
[itk-dev/devops_itkdev-docker](https://github.com/itk-dev/devops_itkdev-docker) instead.

| Workflow         | Checks                                                              |
|------------------|---------------------------------------------------------------------|
| `changelog.yaml` | `CHANGELOG.md` has been updated in the pull request                 |
| `markdown.yaml`  | Markdown files pass `markdownlint` (`.markdownlint.jsonc`)          |
| `lint.yml`       | Lint and format with `ruff`, types with `basedpyright`              |
| `tests.yml`      | `uv.lock` is up to date (`uv lock --check`) and `pytest --cov` runs |
| `yaml.yaml`      | YAML files are formatted with Prettier                              |

## Services

`lint.yml` and `tests.yml` run their checks inside docker compose services. They find the services by
reading `docker-compose.yml` (`docker compose config`) and picking every service with this label:

```yaml
services:
  api:
    labels:
      dk.itkdev.python: "true"
```

Every job runs once per service, and each service shows up as its own check, e.g. `Tests (api)`. If no service
has the label, the workflows fail.

### Override (`PYTHON_SERVICES`)

To choose the services by hand, set the repository variable `PYTHON_SERVICES` to a JSON list of service names.
When it is set, the labels are ignored. Set it under **Settings → Secrets and variables → Actions → Variables**,
or with the [GitHub CLI](https://cli.github.com/):

```shell
# One service
gh variable set PYTHON_SERVICES --body '["api"]'

# More services
gh variable set PYTHON_SERVICES --body '["api", "worker"]'
```

The value must be valid JSON (double quotes). The variable lives in GitHub, not in the repository, so it survives
`template:update`.

## Assumptions

1. `.env.example` exists and contains every variable `docker-compose.yml` requires. The workflows copy it to `.env`.
2. `docker-compose.yml` uses `APP_UID`/`APP_GID` for the container user and an external network named `frontend`;
   both are set up by the workflows.
3. Each Python service (labelled, or listed in `PYTHON_SERVICES`) can run with `--no-deps` and has `uv`, `pytest` with
   [pytest-cov](https://pytest-cov.readthedocs.io/), `ruff` and `basedpyright` available.
4. Coverage is configured in `pyproject.toml` (`[tool.coverage.run] source = [...]`), since the workflow runs
   plain `pytest --cov`.
