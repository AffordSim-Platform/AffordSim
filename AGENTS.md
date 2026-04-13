# Repository Guidelines

## Project Structure & Module Organization
Core Python packages live under `source/`:
- `source/isaaclab`: core simulation and framework code.
- `source/isaaclab_assets`: robot/sensor asset definitions.
- `source/isaaclab_tasks`: task and environment implementations.

Execution entry points are in `scripts/` (RL, imitation learning, demos, benchmarks, tools). Developer utilities are in `tools/` (custom pytest orchestration, templates, helpers). Supporting material is split across `docs/`, `docker/`, `apps/`, and data/asset folders such as `models/`, `scene_reconstruction/`, `backgrounds/`, and `pc_dataset_generate/`.

## Build, Test, and Development Commands
Use `isaaclab.sh` as the primary interface:
- `./isaaclab.sh -c env_isaaclab`: create the conda environment from `environment.yml`.
- `./isaaclab.sh -i`: install editable extensions under `source/` and RL extras.
- `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK> ...`: run project scripts with the correct Python runtime.
- `./isaaclab.sh -t`: run the pytest workflow configured in `tools/conftest.py`.
- `./isaaclab.sh -p tools/run_all_tests.py --discover_only`: list discoverable tests and timeouts.
- `./isaaclab.sh -f`: run repository-wide pre-commit formatting/lint checks.

## Coding Style & Naming Conventions
Target Python 3.11. Follow `pyproject.toml` settings:
- isort line length: 120.
- Import grouping for first-party modules (`isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, etc.).
- Pyright type checking is enabled for `source/` and `scripts/` (`typeCheckingMode = basic`).

Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
Tests are discovered by filename (`test_*.py`) under `source/` and `scripts/`. Keep new tests there so `./isaaclab.sh -t` can find them. Use `@pytest.mark.isaacsim_ci` for CI-scoped tests. If a test needs special timeout or temporary skip handling, update `tools/test_settings.py`.

## Commit & Pull Request Guidelines
Current history is minimal, so use clear imperative commit messages (for example, `Add GO2 terrain randomization config`). Include scope when useful. Contributions should be DCO-compliant; sign commits with `git commit -s`.

For PRs, include:
- What changed and why.
- Exact validation commands run.
- Isaac Sim / Python environment details.
- Screenshots or logs for UI, visualization, or training-behavior changes.

## Collaboration Preferences

### First-Principles Thinking
Use first-principles thinking. Do not assume the user always knows the exact goal or the best path.
Start from the underlying problem and intended outcome.
If the motivation, scope, or target is unclear, stop and clarify with the user before making changes.
If the goal is clear but the path is inefficient, say so directly and suggest a shorter or simpler approach.
Prefer Chinese in normal communication unless the user asks otherwise.

### Communication Style
Keep communication concise.
During normal debugging or investigation, do not narrate every intermediate step unless the user asks for it.
Focus on running checks and tests first, then report the final findings clearly.
Do not use labels such as P0, P1, or P2.
Describe issues in plain language.
When suitable, present results in tables for readability.

## Development Workflow

### Skill-First Execution
At the start of each task, identify the most suitable skill or workflow to use.
Briefly state which skill(s) will be used and how the work will be organized.
At the end, state which skill(s) were used.
