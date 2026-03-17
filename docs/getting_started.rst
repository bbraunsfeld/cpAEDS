Getting Started
===============

This page describes the full cpAEDS workflow, from setting up simulations to
running reweighting analysis.

Installation
------------

Create and activate the conda environment::

    conda env create -f devtools/conda-envs/test_env.yaml
    conda activate constph

Install the package in development mode::

    pip install -e .

Workflow Overview
-----------------

The cpAEDS pipeline consists of five stages:

1. **System setup** — generate GROMOS input files and folder structure
2. **Job submission** — submit MD jobs to a cluster or local machine
3. **Post-processing** — extract energies and calculate free energies
4. **Reweighting analysis** — reconstruct pH curves from a single simulation
5. **Visualisation** — generate publication-quality figures

Stage 1: System Setup
---------------------

Create a ``settings.yaml`` configuration file (see ``cpaeds/tests/test_data/test_settings.yaml``
for an example) and run::

    from cpaeds.system import SetupSystem

    setup = SetupSystem("settings.yaml")
    setup.run_checks()
    setup.create_prod_run()

This creates a folder tree under ``aeds/`` with all required GROMOS input files
(``.imd``, ``.arg``, ``.job``) for each EIR offset point.

Stage 2: Job Submission
-----------------------

Submit jobs using the provided entry-point scripts::

    python -m cpaeds.start_run --config settings.yaml

Jobs are tracked via ``status.yaml`` in the run directory.

Stage 3: Post-Processing
------------------------

After simulations finish, extract energies and compute free energies::

    from cpaeds.postprocessing_parallel import postprocessing_parallel
    import yaml

    with open("settings.yaml") as f:
        config = yaml.safe_load(f)

    pp = postprocessing_parallel(config)
    pp.run_postprocessing_parallel(tasks=4)

This produces ``results/results.out`` (CSV with offsets and fractions) and
``results/energies.npy`` (binary energy array).

Stage 4a: Automatic Linspace Estimation
-----------------------------------------

Before running the reweighting scan you need to choose the ``lnspace`` array —
the range of artificial EIR offset values to scan.  The centre of this range
must coincide with the **equilibrium offset**: the offset at which the system
would show 50/50 protonation/deprotonation.

:func:`~cpaeds.linspace_estimator.auto_linspace` determines this automatically::

    from cpaeds.linspace_estimator import auto_linspace

    est = auto_linspace(
        path="/path/to/run/ene_ana",
        active_state=0,   # index of the protonated state
        n_points=101,     # matching notebook convention
        width=50.0,       # ±25 kJ/mol around equilibrium
    )
    print(f"EIR_eq = {est.center:.1f} kJ/mol  (method: {est.method})")

Three estimation methods are tried in order:

1. **Logistic fit** (``prefer='results'``, default): fits ``results.out`` if
   it exists — most accurate.
2. **Free-energy correction** (``prefer='df'``): reads ``df.out`` and ``e*r.dat``;
   computes ``EIR_eq = EIR_actual + ΔF``.
3. **Actual offset fallback**: uses the raw simulation EIR as the centre.

For multiple statistical runs, :func:`~cpaeds.linspace_estimator.batch_linspace`
estimates each run independently and merges them::

    from cpaeds.linspace_estimator import batch_linspace

    merged, individuals = batch_linspace(
        run_paths=["/path/run1/ene_ana", "/path/run2/ene_ana"],
        active_state=0,
    )
    # merged covers all individual centres

Stage 4: Reweighting Analysis
------------------------------

The reweighting approach allows reconstruction of the pH-dependent protonation
fraction from a **single simulation** by exponential reweighting of ensemble
snapshots to different artificial EIR offset values.

Basic usage::

    import numpy as np
    from cpaeds.reweighting import reweighting_constpH, parallel_reweight

    # Single run, single offset value
    result = reweighting_constpH(
        cutoff=-400.0,        # energy cutoff for A/B classification (kJ/mol)
        temp=300.0,           # temperature (K)
        stepsize=0.5,         # MD output time step (ps)
        itime=4000.0,         # starting time (ps)
        group=[["1"], ["0"]], # state grouping
        path="/path/to/run/ene_ana",
        H1_eds=True,          # use EDS-corrected reference energy
        art_eir=[-220.0, 0.0] # artificial per-state EIR offsets
    )
    print(f"A_frac = {result.A_frac:.3f}")

Scanning over a range of offset values in parallel::

    results = parallel_reweight(
        runs=[1, 2, 3],
        lnspace=np.linspace(-250.0, -190.0, 61),
        nthreads=8,
        path_template="/path/to/aeds/ASPD_{run}/ene_ana",
        cutoff=-400.0,
        temp=300.0,
        stepsize=0.5,
        itime=4000.0,
        group=[["1"], ["0"]],
        H1_eds=True,
    )
    # results[run_index][eir_index].A_frac

Stage 5: Visualisation
-----------------------

Plotting offset vs. fraction from post-processing::

    from cpaeds.plots import StdPlot

    p = StdPlot(basepath="results/", pKa=4.81)
    p.offset_fraction()
    p.offset_pH()

Plotting reweighted titration curves::

    from cpaeds.reweighting_plots import ReweightPlot

    rp = ReweightPlot(
        results=results,
        lnspace=np.linspace(-250.0, -190.0, 61),
        ref_eir=-220.0,
    )
    rp.plot_A_rew()
    rp.plot_titration(pKa=4.81)

Reweighting Method
------------------

The reweighting formula is:

.. math::

   \langle Q \rangle_{\text{rew}} =
   \frac{\langle Q \cdot e^{-\beta(H_i - H_{\text{ref}})} \rangle}
        {\langle e^{-\beta(H_i - H_{\text{ref}})} \rangle}

where:

- :math:`Q` is the observable (binary state indicator 0/1)
- :math:`H_i` is the target Hamiltonian at artificial offset *i*
- :math:`H_{\text{ref}}` is the simulation reference energy (``eds_vr.dat``)
- :math:`\beta = 1 / (k_B T)`

Two reference energy variants are available:

- ``H1_eds=True``: :math:`H_{\text{ref}}` is replaced by the EDS log-sum-exp
  over states: :math:`H_r = -(1/\beta) \ln \sum_s e^{-\beta(H_s - \text{EIR}_s)}`
- ``H1_aeds=True``: same as above but with a quadratic soft-core correction
  in the transition region :math:`[E_{\min}, E_{\max}]`
