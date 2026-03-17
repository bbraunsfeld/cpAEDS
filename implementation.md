# cpAEDS — Reweighting Integration: Implementation Plan

## Context

The cpAEDS package automates constant-pH molecular dynamics (cpH-MD) simulations using the GROMOS engine with AEDS (Alchemical EDS). Originally, independent runs at different EIR offsets were intended to form titration curves. The approach has since shifted to **exponential reweighting** — calculating reweighted protonation fractions for arbitrary offset values from a single simulation, enabling pH-curve reconstruction without running a full titration series.

The reweighting logic currently lives in `Desktop/cluster_scripts/ana/gt_png/reweighing.ipynb`. The goal is to **move this into the cpAEDS package**, make it testable, and document it properly.

---

## Scope

| Area | Status |
|---|---|
| Core reweighting module (`cpaeds/reweighting.py`) | To be created |
| Parallel reweighting runner | To be created |
| Reweighting plots (`cpaeds/reweighting_plots.py`) | To be created |
| Clean analysis example notebook | To be created |
| Tests for reweighting | To be created |
| Tests for existing modules | To be added/improved |
| Documentation (docstrings + Sphinx) | To be updated |
| Conda environment | To be updated |
| Cleanup of stale files | `plot_OR.py`, placeholder `cpaeds.py` |
| Debugging existing modules | As discovered |

---

## Current Codebase Inventory

```
cpaeds/
  algorithms.py       # pKa, offset_steps, logistic fitting
  aeds_sampling.py    # sampling class with Boltzmann averaging
  context_manager.py  # set_directory context manager
  plots.py            # StdPlot with offset/fraction/KDE plots
  plot_OR.py          # OLD duplicate — DELETE
  cpaeds.py           # Empty placeholder — DELETE or repurpose as CLI entry
  file_factory.py     # IMD/job file generation
  system.py           # SetupSystem pipeline
  postprocessing_parallel.py
  pp_run.py, start_run.py, continuous_run.py, check_runs.py
  ssh_connection.py, logger.py, utils.py, cr_pp.py
  tests/
    test_cpaeds.py    # trivial import test
    test_system.py    # setup/folder creation tests
```

**External dependency to absorb:**
- `reweight_by_state.py` (`sampling` class used in notebook via path hack) — confirm if identical to `aeds_sampling.py::sampling`; consolidate.

---

## Architecture of the New Module

### `cpaeds/reweighting.py`

Pure-Python module (no I/O side effects), testable. Implements:

```python
# Core physics
Hr(BETA, energy, offset, avg=True)          # EDS exponential average
Hr_eds(BETA, energy, offset, avg=True)      # Per-state EDS Hr
Hr_s(BETA, energy, offset, emin, emax, avg=True)  # Soft-core corrected Hr

reweighting(Q, H_i, H_ref, beta)            # Exponential reweighting formula

# File I/O helpers (thin wrappers, testable with small fixtures)
read_offset_file(file_name)                 # reads e*r.dat offset files
get_offsets(path=".")                       # collects all offset files in dir
get_eminmax(path=".")                       # reads eds_emax.dat / eds_emin.dat

# Main analysis function
reweighting_constpH(
    cutoff, temp, stepsize, itime, group, path,
    H1_eds=False, H1_aeds=False,
    art_eir=[], art_eminmax=[], gro_rew=False
) -> tuple[mix, reference, A_rew, B_rew, A_frac, B_frac, p_A, p_B, emin, emax]

# Parallel runner
parallel_reweight(runs, lnspace, nthreads, path, **kwargs) -> list[list[tuple]]
```

**Key return type**: each call to `reweighting_constpH` returns a named 10-tuple (consider `dataclasses.dataclass` or `typing.NamedTuple` for clarity):

```python
ReweightResult = NamedTuple('ReweightResult', [
    ('mix', np.ndarray),
    ('reference', np.ndarray),
    ('A_rew', float),
    ('B_rew', float),
    ('A_frac', float),
    ('B_frac', float),
    ('p_A', np.ndarray),
    ('p_B', np.ndarray),
    ('emin', float),
    ('emax', float),
])
```

### `cpaeds/reweighting_plots.py`

Extends `Plot` base class or stands alone. Methods:

```python
class ReweightPlot:
    plot_rew_fraction(results, lnspace, runs, ax=None)      # A_rew vs delta_EIR with std band
    plot_rew_pH(results, lnspace, pKa, runs, ax=None)       # Titration curve from reweighted fractions
    plot_rew_error_bands(results, lnspace, runs, ax=None)   # fill_between std error visualization
```

---

## Implementation Tasks

> Annotation legend:
> - **[H]** → use `claude-haiku-4-5` (routine generation, boilerplate, tests for pure functions)
> - **[S]** → use `claude-sonnet-4-6` (complex algorithm translation, integration, debugging)
> - **[COMPACT]** → compact conversation before this step (context getting large)
> - **[SAVE]** → save key state to memory (important decision / algorithm settled)

---

### Phase 1 — Cleanup

**Task 1.1 [S]**: Investigate `plot_OR.py` — confirm it is a stale duplicate of `plots.py`. If confirmed, delete it.

**Task 1.2 [S]**: Investigate `cpaeds.py` — it's currently a placeholder. Decide: either implement as a CLI entry-point (e.g., using argparse to dispatch `system`, `run`, `postprocess`, `reweight` commands) or remove.  Ask user before deleting or significantly modifying existing code.

**Task 1.3 [S]**: Check if `reweight_by_state.py` on the cluster is identical to `aeds_sampling.py::sampling`. The notebook does:
```python
os.chdir("/pool/bbraun/resources/scripts")
from reweight_by_state import sampling
```
If it is the same class, no new file is needed; the reweighting module should simply import from `cpaeds.aeds_sampling`. If it differs, reconcile or copy.

---

### Phase 2 — Core Reweighting Module

**Task 2.1 [S]**: Create `cpaeds/reweighting.py`:

1. Move/adapt from notebook: `Hr`, `Hr_eds`, `Hr_s`, `reweighting`, `read_offset_file`, `get_offsets`, `get_eminmax`
2. Implement `reweighting_constpH()` — direct translation from notebook, replacing the path-hacked `sampling` import with `from cpaeds.aeds_sampling import sampling`
3. Define `ReweightResult` NamedTuple
4. Implement `parallel_reweight()` using `multiprocessing.Pool`
5. All functions must have complete Google-style docstrings
6. No hardcoded paths — all paths as parameters

> **[SAVE]** after Task 2.1: save to memory that `reweighting.py` was created and depends on `aeds_sampling.sampling.read_energy_file`.

**Task 2.2 [S]**: Update `cpaeds/__init__.py` to expose:
```python
from cpaeds.reweighting import reweighting_constpH, parallel_reweight, ReweightResult
```

**Task 2.3 [S]**: Update `environment.yml` and `devtools/conda-envs/test_env.yaml`:
- Ensure `natsort` is present (used in reweighting)
- Ensure `scipy` version supports `curve_fit` with `dogbox` method
- Add `jupyter` and `ipykernel` to docs/examples env if not present

---

### Phase 3 — Reweighting Plots

**Task 3.1 [S]**: Create `cpaeds/reweighting_plots.py`:

1. `ReweightPlot` class that accepts the output of `parallel_reweight()` (nested list of `ReweightResult`)
2. `plot_A_rew_vs_delta_eir(results, lnspace, runs, ax=None)`:
   - x-axis: delta_EIR (lnspace - art_eir_reference)
   - y-axis: A_rew values (index [2] from results)
   - std dev band from multiple runs using `fill_between`
3. `plot_titration_curve(results, lnspace, pKa, runs, ax=None)`:
   - Converts A_frac to pH via `ph_curve()` from `algorithms.py`
   - Plot with error bands

---

### Phase 4 — Tests

**Task 4.1 [H]**: Create `cpaeds/tests/test_reweighting.py`:

Cover:
- `Hr()`: verify against analytically computed exponential average for small arrays
- `Hr_s()`: verify the three branches (above emax, between, below emin)
- `reweighting()`: verify that with uniform Q the reweighted result equals Q
- `reweighting()`: verify normalization (sum of p ≈ n_frames)
- `read_offset_file()`: create a tiny temp file with known content, assert correct float returned
- `get_offsets()`: create temp dir with 2 offset files, assert list of 2 floats
- `reweighting_constpH()`: create minimal synthetic energy files and assert `A_frac + B_frac ≈ 1`

> Test fixtures: small synthetic `.dat` files (3–5 frames, 2 states) placed in `cpaeds/tests/test_data/reweighting/`

**Task 4.2 [H]**: Extend `test_system.py`:
- Add test for `cpAEDS_type = 3` (currently untested)
- Add test for `cpAEDS_type = 4`

**Task 4.3 [H]**: Create `cpaeds/tests/test_algorithms.py`:
- Test `pKa_from_df()` with known values (e.g., df=0 → pKa depends only on T)
- Test `ph_curve()` at f=0.5 → pH == pKa
- Test `logistic_curve()` and `inverse_log_curve()` are inverses
- Test `log_fit()` recovers known parameters for synthetic data

**Task 4.4 [H]**: Create `cpaeds/tests/test_sampling.py`:
- Test `sampling.read_energy_file()` with a synthetic file
- Test `sampling.main()` with minimal synthetic energy files

> **[COMPACT]** before Phase 4: conversation will be large; compact and give summary to new context.

---

### Phase 5 — Documentation

**Task 5.1 [S]**: Add/update Google-style docstrings to all public functions in:
- `reweighting.py` (done in Phase 2)
- `algorithms.py` (many missing or incomplete)
- `aeds_sampling.py`
- `plots.py`

**Task 5.2 [H]**: Update `docs/api.rst` to include `cpaeds.reweighting` and `cpaeds.reweighting_plots` modules.

**Task 5.3 [H]**: Update `docs/getting_started.rst` — add a "Reweighting Analysis" section describing the new workflow:
1. Run simulations with `system.py` + `start_run.py`
2. Post-process with `postprocessing_parallel.py`
3. Run reweighting with `reweighting.py`
4. Visualize with `reweighting_plots.py`

**Task 5.4 [H]**: Update `README.md` — add one-paragraph description of the reweighting workflow.

---

### Phase 6 — Example Notebook

**Task 6.1 [S]**: Create `examples/reweighting_analysis.ipynb`:

A clean, well-documented notebook that:
1. Shows how to import and call `parallel_reweight()`
2. Shows how to load and inspect `ReweightResult` objects
3. Shows how to use `ReweightPlot` to generate publication-quality figures
4. Uses synthetic/demo data (no hardcoded cluster paths)

This replaces the `reweighing.ipynb` for new users — the original notebook can be archived.

---

### Phase 7 — Debugging Pass

**Task 7.1 [S]**: Run the full test suite (`pytest -v`) and fix any failures.

Known potential issues to check:
- `plots.py:165` has a syntax error (indentation off after `set_ylabel`): `ax.set_xlabel` is at wrong indentation level under `if diff == True:` block — verify and fix.
- `aeds_sampling.py:270–275`: `isclose` tolerance check between `tot_con_frames` and `tot_concut_frames` may fail for multi-state systems if groupB frames are large — verify logic is correct.
- `algorithms.py:48`: `offsets[EIR_groups[-1][-1]]` uses the last element of the last group as reference for `n_offsets` — confirm this is correct for all cpAEDS_type cases.

**Task 7.2 [S]**: Check `postprocessing_parallel.py` for any silent failures (unhandled exceptions in parallel workers).

---

### Phase 8 — Integration Check

**Task 8.1 [S]**: Run `pytest --cov=cpaeds --cov-report=term-missing` and target ≥ 60% coverage on the new modules.

**Task 8.2 [S]**: Verify `environment.yml` creates a working conda env:
```bash
conda env create -f environment.yml
conda activate cpaeds
pip install -e .
pytest cpaeds/tests/
```

---

## File Changes Summary

| File | Action |
|---|---|
| `cpaeds/reweighting.py` | **CREATE** |
| `cpaeds/reweighting_plots.py` | **CREATE** |
| `cpaeds/tests/test_reweighting.py` | **CREATE** |
| `cpaeds/tests/test_algorithms.py` | **CREATE** |
| `cpaeds/tests/test_sampling.py` | **CREATE** |
| `cpaeds/tests/test_data/reweighting/` | **CREATE** (synthetic fixture files) |
| `examples/reweighting_analysis.ipynb` | **CREATE** |
| `cpaeds/__init__.py` | **MODIFY** (add reweighting exports) |
| `environment.yml` | **MODIFY** (add natsort if missing) |
| `devtools/conda-envs/test_env.yaml` | **MODIFY** (sync with environment.yml) |
| `docs/api.rst` | **MODIFY** |
| `docs/getting_started.rst` | **MODIFY** |
| `README.md` | **MODIFY** |
| `cpaeds/plot_OR.py` | **DELETE** (after confirmation) |
| `cpaeds/cpaeds.py` | **DECIDE** (repurpose or delete — ask user) |
| All existing `.py` modules | **DOCSTRINGS ONLY** (no logic changes) |

---

## Token Efficiency Strategy

| When | Action |
|---|---|
| After Phase 1 cleanup confirmed | Save memory: "plot_OR.py deleted, cpaeds.py decision: <X>" |
| After `reweighting.py` is finalized | Save memory: module created, key algorithms, NamedTuple shape |
| Before Phase 4 (tests) | **COMPACT** — tests are mechanical, fresh context is more efficient |
| After test suite passes | Save memory: test coverage %, known fixture structure |
| Before Phase 5 (docs) | **COMPACT** if context is large |

### Model Selection

| Task | Model | Reason |
|---|---|---|
| Algorithm translation (reweighting.py) | **Sonnet** | Complex physics logic, careful translation |
| Integration / imports / __init__ | **Sonnet** | Cross-file reasoning needed |
| Cleanup decisions (plot_OR, cpaeds.py) | **Sonnet** | Needs code comprehension |
| Test boilerplate for pure functions | **Haiku** | Mechanical, repetitive |
| Docstring writing | **Haiku** | Template-driven |
| Sphinx rst updates | **Haiku** | Template-driven |
| Debugging (syntax errors, logic bugs) | **Sonnet** | Needs reasoning |
| Full integration run / coverage check | **Sonnet** | Multi-step orchestration |

---

## Resolved Decisions

1. **`reweight_by_state.py`** — found locally at `Desktop/cluster_scripts/resources/scripts/aeds_helper/reweight_by_state.py`. It is a *different* class from `aeds_sampling.py::sampling` (different constructor, adds `mixing_by_states`, `write_group_mixing`, `write_group_diff`). The notebook only calls the identical `read_energy_file()` static method. Plan: absorb the unique `mixing_by_states` logic into the package alongside the reweighting module (either extend `aeds_sampling.py` or fold into `reweighting.py`).

2. **`cpaeds.py`** — delete it. Other entry points (`pp_run.py`, `start_run.py`, etc.) cover the use cases.

3. **`reweighing.ipynb`** — archive to `examples/archive/reweighing.ipynb`.

4. **Python version** — intended 3.10+, but the available system Python is **3.9.5** and all conda envs have broken numpy DLLs. Code targets 3.9 for compatibility. `match`/`case` replaced with `if/elif/else`.
