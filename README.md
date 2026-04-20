# dep_extractor

> **ComfyUI Dependency Resolver Suite — v2.0**

A modular, concurrent Python package that replaces the original 1,085-line monolithic `requirements_extractor.py`.

## Why this repo exists

The legacy requirement extractor was tightly coupled to the StabilityMatrix ComfyUI installation at `D:\Downloads\StabilityMatrix-win-x64\...`, making it difficult to develop, test, and distribute independently. This repo is the clean, standalone extraction.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **Node scanning** | Recursively discovers all `requirements.txt` files in custom_nodes |
| 🖥️ **Hardware audit** | Probes CUDA version, GPU VRAM, Windows Insider channel, uv availability |
| ⚡ **Static conflict detection** | Flags version pin conflicts before any install attempt |
| 🔬 **SAT simulation** | `uv pip compile` dry-run to validate full dependency resolution |
| 🌐 **Concurrent URL validation** | `httpx` + `asyncio` — validates all online deps in parallel (~5s vs ~30s serial) |
| 📄 **Report writer** | Generates `combined_requirements.txt` with index URLs, sorted packages, and a diagnostic audit section |
| ☢️ **Heavy compiler support** | MSVC toolchain detection (`vswhere.exe` + `vcvarsall.bat`) and phased build for deepspeed/flash-attn/xformers |
| 🎨 **Rich terminal UI** | ADHD-forward phase panels, progress bars, color-coded conflict tables |

---

## Package Structure

```
dep_extractor/
├── models.py           # Typed dataclasses (the shared data contracts)
├── config.py           # Constants registry (heavy compilers, platform remaps, CUDA indices)
├── core/
│   ├── normalizer.py   # Unified PEP 508 requirement line parser
│   ├── scanner.py      # Recursive node/requirements discovery
│   ├── auditor.py      # Hardware environment probe
│   └── simulator.py    # uv SAT-solver wrapper (dry-run)
├── analysis/
│   ├── conflict_detector.py  # Static pre-resolution conflict analysis
│   ├── url_validator.py      # Concurrent async URL health checks
│   └── wheel_tracker.py      # Local .whl file cross-reference
├── output/
│   ├── console.py      # All terminal output (Rich — single Console instance)
│   └── report_writer.py # combined_requirements.txt writer (atomic writes)
├── install/
│   ├── toolchain.py    # MSVC/Ninja/CMake detection
│   ├── heavy_compiler.py # Heavy package build orchestration
│   └── installer.py    # Phased installation orchestrator
└── cli.py              # Thin argparse entry point
```

---

## Installation

```powershell
# Into a virtual environment
uv pip install -e .

# With dev/test dependencies
uv pip install -e ".[dev]"
```

---

## Usage

```powershell
# Scan, analyze, write combined_requirements.txt
dep-extractor --root "D:\path\to\custom_nodes"

# Skip URL validation (faster, offline)
dep-extractor --root . --quick

# Full pipeline with SAT simulation + hardware audit display
dep-extractor --root . --simulate --audit

# Generate report only (no install)
dep-extractor --root . --no-install --output my_requirements.txt
```

---

## Testing

```powershell
# Run all tests (28 tests, ~0.1s)
pwsh -File run_tests.ps1

# Or directly
python -m pytest tests/ -v
```

**Coverage**: normalizer (12), conflict_detector (11), report_writer (5)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `rich>=13.0` | Terminal UI (panels, tables, progress bars) |
| `httpx>=0.25` | Concurrent async HTTP for URL validation |
| `packaging>=23.0` | PEP 440 version specifier comparison |

Optional runtime tools (not Python packages):
- `uv` — fast package installer and SAT solver
- `pwsh` (PowerShell 7+) — used for hardware probing

---

## Architecture Principles

1. **Single responsibility** — every module has one job
2. **EAFP over LBYL** — ask forgiveness, not permission (no crashes on unusual environments)
3. **Single `sys.exit()`** — only in `cli.py`, never inside library code
4. **Atomic file writes** — `tmp → rename` in `report_writer.py`
5. **Probed once** — toolchain detection runs once per session, result passed to all callers
6. **No monolith modification** — the original `requirements_extractor.py` is untouched

---

## License

MIT
