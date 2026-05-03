# KAN Dispersion-to-RL Demo

Week 1 controlled-physics demo for the upgraded pipeline:

```text
material features -> TinyKAN -> Cole-Cole parameters -> eps/mu(f) -> transmission-line RL(f,d)
```

The neural net does not learn frequency response directly. It only maps material features to physically constrained dispersion parameters; Cole-Cole and transmission-line equations generate `eps(f)`, `mu(f)`, and reflection loss.

## Headline Result

On synthetic material data with a material-level train/validation split:

```text
TinyKAN      : 0.115 +/- 0.020 validation loss
Matched MLP  : 0.733 +/- 0.033 validation loss
Params       : 737 vs 748, 1.49% gap
Seeds        : 7, 11, 19
```

The loss gap is a sanity check, not the main claim. The useful result is that TinyKAN's edge functions visibly recover controlled non-monotonic mechanisms, such as annealing-temperature windows and carbon-ratio relaxation effects.

## Why This Demo Exists

This is a proof-of-chain before touching noisy literature data:

- Can a small KAN-style model recover material -> dispersion mappings?
- Can the differentiable physics layer safely propagate through Cole-Cole, complex `eps/mu`, and RL?
- Can edge functions serve as a hypothesis generator rather than just another prediction head?

## Run

From the repository root:

```powershell
python .\run_demo.py
```

Fast smoke test:

```powershell
python .\run_demo.py --quick
```

`--quick` uses `n_materials=32`, `epochs=20`, and `n_freq=16`. It checks forward/backward/file IO only; plots are not expected to be presentation quality.

Recommended pre-meeting run:

```powershell
python .\run_demo.py --epochs 300 --n-materials 256 --seeds 7 11 19
```

Optional wider MLP reference:

```powershell
python .\run_demo.py --include-width-mlp
```

## Outputs

Generated files are written under `outputs/`:

- `summary.md`: short presentation-ready summary.
- `metrics.json`: full per-seed, per-dimension metrics.
- `kan_edge_functions.png`: TinyKAN edges vs known synthetic partial effects.
- `eps_mu_decomposition.png`: predicted vs true `eps'`, `eps''`, `mu'`, `mu''`.
- `rl_curve.png`: RL curve for one held-out material.
- `rl_heatmap.png`: thickness-frequency RL map.
- `param_scatter.png`: predicted vs true dispersion parameters.
- `edge_l1_scores.csv` and `edge_l1_scores.png`: RBF-edge L1 scores.

## What This Demonstrates

- Frequency is handled by physics, not learned directly by the network.
- Cole-Cole uses polar form instead of unstable complex fractional powers.
- Output constraints enforce positive `tau`, bounded `alpha`, and `eps_s > eps_inf`.
- Parameter losses are standardized per dimension, so `tau`, `alpha`, and permittivity do not dominate each other by scale.
- RL loss is only a `0.1x` consistency term; parameter and `eps/mu` supervision are dominant.
- TinyKAN and MLP share the same input normalization, optimizer, schedule, and `PhysicalOutputHead`.
- Validation is split by material ID, not by individual frequency points, so there is no frequency-point leakage.
- Edge plots use row-level scaling, so weak/dummy edges are not independently amplified.

## What This Does Not Demonstrate

- It does not prove automatic pruning. The current active/dummy edge L1 ratio is `1.79x`, which is weak evidence only.
- It does not perform symbolic regression. This local TinyKAN is an RBF-edge stand-in, not `pykan`.
- It does not use real literature data. All data are synthetic Cole-Cole samples with known ground truth.
- It does not solve real-data Kramers-Kronig consistency. Literature `eps/mu` must later be fitted jointly in the complex domain.
- It does not make the Week 1 dashed ground-truth edge plot available for real data. Week 3 should use edge functions with bootstrap confidence intervals.

## Current Limitations

- `pykan` pruning and `auto_symbolic()` are deferred to Week 2.
- If `pykan` is added, inputs should be exported on `[-1, 1]`, symbolic basis functions should be physically restricted, and L1 regularization should be used.
- Literature curation should start with one controlled material family and SEM/TEM evidence that morphology does not jump across the series.
- Carbon-dominant materials may need a Debye + Drude term; this demo only uses Cole-Cole.

## Tests

```powershell
python -m py_compile .\run_demo.py .\physics.py .\test_physics.py
python -m unittest .\test_physics.py
python .\run_demo.py --quick
```

## 30-Second Pitch

I am not using KAN as a black-box RL predictor. The network only maps material features to constrained Cole-Cole dispersion parameters; analytic physics generates `eps(f)`, `mu(f)`, and RL. In this controlled Week 1 test, the KAN-style edge structure recovers non-monotonic mechanisms such as annealing-temperature windows under a very small parameter budget. The loss gap is only a sanity check; the real point is that KAN gives inspectable edge functions that can become hypotheses for material mechanisms. Symbolic `pykan` and literature data are the next milestone, not this demo.
