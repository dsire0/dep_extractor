# Project Instructions: dep_extractor

## Tech Stack
- **Language**: Python 3.12+ 
- **Type**: Standalone CLI Package (extracted from StabilityMatrix / ComfyUI ecosystem)
- **Packaging**: uv & pyproject.toml (Setuptools)

## Code Style
- **Naming Conventions**: Standard PEP 8 (snake_case for variables/functions, PascalCase for classes). Internal helper functions are prefixed with `_` (e.g., `_should_skip_dir`).
- **Patterns**:
  - Modular, single-responsibility layers (e.g. `core`, `analysis`, `install`, `output`).
  - Clear, educational docstrings.
  - Fail-safe operations (e.g. `read_file_safe` tries multiple encodings and falls back to empty strings without crashing).
  - UI output handled cleanly by the `output/console.py` layer using Rich.

## Testing
- Run tests: `.\run_tests.ps1` (runs pytest)
- Test pattern: Tests are located in `tests/test_*.py`
- Code should be tested locally after any changes.

## Build & Run
- Dev Install: `uv pip install -e .` (Add `--system` if in miniconda base).
- Use: `dep-extractor --root <path>`
- Root Entrypoint: `dep_extractor/cli.py`

## Project Structure
- `dep_extractor/` - Main package
  - `core/` - normalizer, scanner, auditor, simulator
  - `analysis/` - conflict_detector, url_validator, wheel_tracker
  - `install/` - toolchain, heavy_compiler, installer
  - `output/` - console, report_writer
  - `cli.py` - main entrypoint
- `tests/` - pytest test suite

## Conventions
- **Issue Tracking**: Mandatory use of `bd` (beads) for issue tracking. Do NOT use markdown TODOs.
- **Commits**: Conventional Commits (e.g., `feat(module): message`, `fix(scanner): ...`).
- **Development Loop**:
  1. `bd ready` logic and issue tracking.
  2. Implement fix.
  3. Run `pytest` via `run_tests.ps1`.
  4. End session: `git pull --rebase && bd dolt push && git push`.
