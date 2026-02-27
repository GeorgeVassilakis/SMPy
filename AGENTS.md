# AGENTS.md

## Cursor Cloud specific instructions

**SMPy** is a pure Python scientific library for weak gravitational lensing mass reconstruction. There are no external services, databases, or Docker containers required.

### Running tests

```bash
python3 -m pytest tests/ -v
```

All 40 tests should pass. The 2 RuntimeWarnings from `ks_plus.py` (divide by zero in Fourier space) are expected and harmless.

### Linting

No project-specific linter config exists. Use flake8 with relaxed line length:

```bash
python3 -m flake8 smpy/ --max-line-length=120
```

Pre-existing style warnings are expected; the project does not enforce flake8 in CI.

### Running the application

- **Python API** (see `README.md` "Quickstart" section for full examples):
  ```python
  from smpy import map_kaiser_squires
  result = map_kaiser_squires(data="examples/data/forecast_lum_annular.fits",
                              coord_system="radec", pixel_scale=0.4,
                              g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight")
  ```
- **CLI runner**: `python3 runner.py -c smpy/configs/example_config.yaml`
  - The example config has hardcoded absolute paths; update `input_path` and `output_directory` before running.

### Non-obvious notes

- Use `python3` (not `python`) — the VM does not have a `python` symlink.
- Set `MPLBACKEND=Agg` when running headlessly to avoid matplotlib display errors.
- Example FITS data files are in `examples/data/` (not tracked by Git LFS, committed directly).
- The package is installed in editable mode (`pip install -e .`), so source changes are picked up immediately.
