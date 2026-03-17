cpAEDS
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/cpAEDS/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/cpAEDS/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/cpAEDS/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/cpAEDS/branch/master)


Pipeline tool to set up constant pH simulations with Gromos using AEDS.

cpAEDS automates the complete constant-pH molecular dynamics workflow:
setup of GROMOS AEDS input files, job submission, parallel post-processing,
and **exponential reweighting** for pH-curve reconstruction from single
simulations without running a full titration series.

### Workflow

```
settings.yaml → SetupSystem → GROMOS MD → postprocessing_parallel → reweighting → plots
```

1. **Setup**: generate per-offset run folders and GROMOS input files
2. **Run**: submit SLURM/bash jobs; monitor with `check_runs`
3. **Post-process**: extract energies and free energies in parallel
4. **Reweight**: reconstruct protonation fractions at arbitrary EIR offsets
   using `reweighting_constpH` / `parallel_reweight`
5. **Plot**: `StdPlot` for offset-vs-fraction, `ReweightPlot` for titration curves

See `docs/getting_started.rst` for a full usage guide.

### Copyright

Copyright (c) 2022, Benedict Braunsfeld


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
