# PSO--KMeans Niching for Multimodal Optimization

Particle Swarm Optimization with clustering-based niching for recovering multiple basin representatives, implemented on Apple Metal for M-series GPUs.

**Manuscript:** *PSO--KMeans Niching for Multimodal Optimization with Metal-GPU Realization on Apple M-Series*

**Authors:**

- Ankit Ruhi, Assistant Professor, Dr. Harisingh Gour University, Sagar, Madhya Pradesh, India
- Ramesh Kolluru, CFD Consultant, Independent Researcher

## Project Overview

This project is a coverage-first empirical study of fixed-$K$ PSO niching with a single-pass Lloyd clustering step over particle-best positions, per-cluster best attractors, and weak-cluster reseeding. The primary contribution is engineering and empirical: a reproducible niching recipe, a Metal GPU realization, and a coverage-centric evaluation protocol. The work does not claim fundamentally new PSO dynamics; algorithmic novelty is limited to the reseeding and cluster-best coupling choices.

**Key features:**

- Metal GPU compute kernels (2D and N-dimensional)
- CUDA GPU kernels (2D and N-D; NVIDIA systems)
- Fixed-$K$ niching (recommended default) plus exploratory enhanced mode (adaptive-$K$ + local refinement)
- Ring/lbest topology baseline on Metal (`--mode ring_lbest`)
- Coverage-centric metrics: BestCov, CentroidCov, MeanDist, MeanDist(repr), collapse diagnostics
- Reviewer-gap studies: CPU ablations, GPU ring comparison, nozzle/diffuser surrogate

---

## Repository Structure

The implementation lives under `Shaders/` in the full research workspace:

```
Shaders/
├── src/
│   ├── gpu/
│   │   ├── PSO.metal
│   │   ├── PSO.cu
│   │   ├── cuda_pso_runner.cu
│   │   ├── metal_gpu_runner.swift
│   │   └── metal_gpu_runner_highdim.swift
│   └── python/
│       ├── run_review_enhancements.py
│       ├── run_reviewer_gap_study.py
│       ├── run_gpu_ring_comparison.py
│       ├── run_nozzle_surrogate.py
│       ├── plot_nozzle_results.py
│       ├── run_highdim_metal_validation.py
│       └── plot_gpu_outputs.py
├── build/
├── docs/
│   ├── manuscripts/
│   │   ├── gpu_pso_multimodal_journal.tex
│   │   └── gpu_pso_multimodal_journal.pdf
│   └── references/
│       └── journal_references.bib
└── results/
    ├── paper_results/
    └── plots/
```

---

## Quick Start

### 1. Compile GPU runners

From the `Shaders` directory:

```bash
mkdir -p build
cd src/gpu
xcrun swiftc metal_gpu_runner.swift -framework Metal -o ../../build/metal_gpu_runner
xcrun swiftc metal_gpu_runner_highdim.swift -framework Metal -o ../../build/metal_gpu_runner_highdim
cd ../..
```

If `xcrun` fails because the Xcode license has not been accepted, complete Apple's license acceptance locally and rebuild.

**CUDA (NVIDIA):**

```bash
cd src/gpu
nvcc -O3 -std=c++17 cuda_pso_runner.cu -o ../../build/cuda_pso_runner
cd ../..
./build/cuda_pso_runner --objective 0 --particles 2048 --iterations 500
```

Objective codes: `0=sphere`, `1=rastrigin`, `2=ackley`, `3=himmelblau`, `4=six-hump`, `5=holder`.

### 2. Run 2D benchmarks

```bash
./build/metal_gpu_runner --mode plain
./build/metal_gpu_runner --mode kmeans
./build/metal_gpu_runner --mode ring_lbest
./build/metal_gpu_runner --mode kmeans --enable-adaptive-k --enable-refinement
```

### 3. Run analysis scripts

```bash
python src/python/run_review_enhancements.py --trials 20
python src/python/run_reviewer_gap_study.py
python src/python/run_gpu_ring_comparison.py
python src/python/run_nozzle_surrogate.py
python src/python/plot_nozzle_results.py
```

### 4. Rebuild the manuscript PDF

```bash
cd docs/manuscripts
pdflatex -interaction=nonstopmode gpu_pso_multimodal_journal.tex
bibtex gpu_pso_multimodal_journal
pdflatex -interaction=nonstopmode gpu_pso_multimodal_journal.tex
pdflatex -interaction=nonstopmode gpu_pso_multimodal_journal.tex
```

---

## Algorithm Summary

Each PSO iteration uses standard velocity/position updates. The social attractor is:

- **Plain mode:** global best
- **Niching mode:** cluster-best for the particle's assigned niche
- **Ring/lbest mode:** neighborhood best on a ring topology

Clustering is a **single-pass Lloyd update** (one assignment plus centroid recompute per PSO iteration), not full K-means to convergence. Clustering runs on particle-best positions; weak clusters are reseeded to farthest particle-best candidates to preserve diversity.

The Metal pipeline keeps **particle state GPU-resident** between phases; **centroid reduction is host-assisted** for deterministic, reproducible results (avoiding nondeterministic GPU atomics).

---

## Evaluation Metrics

| Metric | Role |
|--------|------|
| **BestCov** | Primary: fraction of known optima with a returned cluster-best within $\epsilon$ |
| **CentroidCov** | Diagnostic: centroid-based coverage |
| **MeanDist** | Diagnostic: mean centroid-to-nearest-optimum distance |
| **MeanDist(repr)** | Mean distance of returned cluster-best representatives to nearest optima |
| **Largest-cluster fraction / entropy** | Collapse diagnostics |

For objectives without known minima, use returned-set clustering validity, archive spread, $\epsilon$-distinct solution count, and minimum separation between representatives.

---

## Main Findings

- Fixed-$K$ clustered PSO improves multimodal recovery over plain PSO on the main 2D benchmark suite (Himmelblau, Six-Hump Camel, Hölder Table, etc.).
- **Enhanced mode** (adaptive-$K$ + refinement) is a **tradeoff mode**: useful in selected cases but not uniformly better than fixed-$K$.
- On simple two-basin geometry (Six-Hump Camel), ring/lbest can be highly competitive; clustering helps more as basin count and geometry complexity increase.
- Hölder Table illustrates why BestCov (representatives) and CentroidCov (centroids) must be reported separately.
- Comparisons are scoped to **within the PSO family on Metal** (plain, fixed-$K$, enhanced, ring/lbest).

---

## Command-Line Reference (2D runner)

| Flag | Description |
|------|-------------|
| `--mode` | `plain`, `kmeans`, or `ring_lbest` |
| `--objective` | Objective name or `all` |
| `--particles` | Swarm size (default: 256) |
| `--iterations` | PSO iterations (default: 500) |
| `--seed` | Random seed |
| `--enable-adaptive-k` | Enable adaptive-$K$ (enhanced / tradeoff mode) |
| `--enable-refinement` | Post-run local centroid refinement |

---

## Documentation

- **`docs/manuscripts/gpu_pso_multimodal_journal.pdf`**: Primary journal manuscript (25 pages)
- **`docs/manuscripts/gpu_pso_multimodal_journal.tex`**: LaTeX source
- **`docs/references/journal_references.bib`**: Bibliography

Reviewer-gap artifacts:

- `src/python/run_reviewer_gap_study.py` — ring/lbest baseline, ablations, coupling sweep
- `src/python/run_gpu_ring_comparison.py` — GPU plain vs K-means vs ring/lbest
- `results/paper_results/reviewer_gap_study_*.csv` — aggregated trial outputs

---

## Requirements

- Apple Silicon Mac (M1/M2/M3 or later recommended) for Metal
- NVIDIA GPU + CUDA toolkit (`nvcc`) for CUDA kernels
- macOS 12.0+ (Metal path)
- Xcode command-line tools (Metal path)
- Python 3.8+ (`numpy`, `pandas`, `matplotlib`, `scipy`) for analysis scripts in the full workspace
- LaTeX (optional, for rebuilding the PDF)

---

## Scope and Limitations

- Strongest evidence is on 2D multimodal benchmarks with known optima; 5D/10D tests are limited validation.
- GPU/CPU timing is feasibility and mapping evidence, not a claim of universal hardware superiority.
- This repository includes CUDA kernels (`PSO.cu`) and a minimal CUDA runner; full CUDA orchestration (K-means host loop, multimodal metrics) is still in progress.
- Canonical niching baselines beyond ring/lbest (e.g., SPSO, clearing, fitness sharing) are not yet included.

---

## Citation

```bibtex
@article{RuhiKolluru2026PSOKMeansMetal,
  title  = {PSO--KMeans Niching for Multimodal Optimization with Metal-GPU Realization on Apple M-Series},
  author = {Ruhi, Ankit and Kolluru, Ramesh},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

---

## License

MIT License. See individual source files for license headers.

---

## Project Status

Manuscript and PDF updated: **March 3, 2026**. CUDA kernels added: **June 2026**.

Repository: [https://github.com/rameshkolluru43/PSO](https://github.com/rameshkolluru43/PSO)
