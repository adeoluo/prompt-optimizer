# AGENTS.md

Working notes for coding agents and contributors in **Prompt Optimizer**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Automated prompt and few-shot optimization built from scratch: compile a prompt against a metric by searching over instructions and labeled demonstrations.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/prompt_optimizer/__init__.py`: Prompt Optimizer - automated prompt and few-shot optimization from scratch.
- `src/prompt_optimizer/cli.py`: Command line interface for the prompt optimizer.
- `src/prompt_optimizer/data.py`: A bundled intent-classification task used for demonstrations and evaluation.
- `src/prompt_optimizer/evaluation.py`: An ablation that measures what each optimizer contributes.
- `src/prompt_optimizer/lm.py`: A deterministic, offline simulated language model.
- `src/prompt_optimizer/metrics.py`: Evaluation metrics and dataset scoring helpers.
- `src/prompt_optimizer/observability.py`: In-process request metrics for the HTTP service.
- `src/prompt_optimizer/optimizers.py`: Optimizers that compile a prompt against a metric.
- `src/prompt_optimizer/predictor.py`: A predictor binds a prompt to the simulated language model.
- `src/prompt_optimizer/service.py`: A standard-library HTTP service exposing the compiled prompt.
- `src/prompt_optimizer/text.py`: Deterministic text utilities shared by the language model and metrics.
- `src/prompt_optimizer/types.py`: Core data types shared across the package.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the application entrypoint |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
