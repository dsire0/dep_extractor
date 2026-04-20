$ErrorActionPreference = 'Stop'
$python = 'd:\Downloads\StabilityMatrix-win-x64\Data\Packages\ComfyUI_RTX\venv\Scripts\python.exe'
Set-Location 'd:\Downloads\StabilityMatrix-win-x64\Data\Packages\ComfyUI_RTX\custom_nodes'
uv pip install --python $python --quiet -e '.[dev]'
& $python -m pytest tests/ -v --tb=short
