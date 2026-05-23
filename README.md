# PSO--KMeans Niching on Apple Metal GPU

## Project Overview

This project primarily investigates a coverage-driven empirical study of a PSO--KMeans niching scheme for multimodal basin recovery, with Metal-GPU realization on Apple M-series hardware as the practical implementation vehicle.

**Key Features:**
- Metal GPU compute kernels (2D and N-dimensional support)
- PSO--KMeans niching strategy for basin-level diversity
- Adaptive-$K$ clustering and local centroid refinement
- Objective-aware adaptive policy thresholds
- Comprehensive evaluation suite with robustness and scaling studies

---

## Repository Structure

```
PSO_CPP/Shaders/
├── src/                              # Source code
│   ├── gpu/                          # GPU compute and orchestration
│   │   ├── PSO.metal                 # Metal compute kernels (2D and N-D)
│   │   ├── metal_gpu_runner.swift    # 2D GPU orchestrator
│   │   └── metal_gpu_runner_highdim.swift  # N-D GPU orchestrator
│   ├── python/                       # Python analysis and visualization
│   │   ├── run_highdim_metal_validation.py   # High-dimensional validation sweep
│   │   ├── run_application_surrogate.py      # Application utility metrics
│   │   ├── run_comparative_study.py          # Baseline comparisons
│   │   ├── run_extended_paper_experiments.py # Extended experiments
│   │   ├── run_review_enhancements.py        # Enhanced mode robustness (20 trials)
│   │   ├── run_highdim_pilot.py              # Pilot high-D runs
│   │   ├── pso_kmeans_bench.py               # Benchmarking utilities
│   │   ├── plot_gpu_outputs.py               # Plot generation
│   │   ├── plot_objective_functions.py       # Objective landscape plots
│   │   ├── plot_plain_vs_kmeans.py           # Comparison visualizations
│   │   └── plot_himmelblau_sequence.py       # Sequence evolution plots
│   └── legacy/
│       └── PSO_OpenCL.cl             # Historical OpenCL version (reference)
├── build/                            # Compiled binaries
│   ├── metal_gpu_runner              # 2D GPU executable
│   └── metal_gpu_runner_highdim      # N-D GPU executable
├── docs/                             # Documentation and manuscripts
│   ├── manuscripts/                  # TeX source and PDFs
│   │   ├── gpu_pso_multimodal_journal.tex       # Primary journal (31 KB)
│   │   ├── gpu_pso_multimodal_journal.pdf       # Journal PDF (15 pages, 3.6 MB)
│   │   ├── gpu_pso_multimodal_documentation.tex # Practical guide (13 KB)
│   │   └── gpu_pso_multimodal_documentation.pdf # Guide PDF (7 pages, 181 KB)
│   ├── references/                   # Bibliography
│   │   └── journal_references.bib
│   ├── CONSOLIDATION_SUMMARY.md      # Manuscript consolidation notes
│   └── README.md                     # This file
├── results/                          # Experimental results and outputs
│   ├── paper_results/                # Final validation CSVs
│   │   ├── highdim_metal_summary.csv
│   │   ├── highdim_metal_trials.csv
│   │   ├── application_surrogate_summary.csv
│   │   ├── application_surrogate_trials.csv
│   │   ├── review_summary.csv
│   │   └── review_trials.csv
│   ├── gpu_outputs/                  # 2D benchmark outputs (plain mode)
│   ├── gpu_outputs_plain/            # 2D plain PSO outputs
│   ├── comparisons/                  # Comparative study outputs
│   │   ├── gpu_compare_kmeans/
│   │   ├── gpu_compare_kmeans_six/
│   │   ├── gpu_compare_plain/
│   │   └── gpu_compare_plain_six/
│   └── plots/                        # Generated figures (PNG)
├── tmp/                              # Temporary execution directories
│   ├── tmp_hd_*/                     # High-dimensional validation trials
│   ├── tmp_review_*/                 # Enhanced mode robustness trials
│   ├── tmp_scaling_*/                # GPU scaling study trials
│   └── tmp_trials_*/                 # Miscellaneous trials
├── latex_aux/                        # LaTeX auxiliary files (.aux, .log, .bbl)
├── archived_manuscripts/             # Previous manuscript versions (for reference)
│   ├── gpu_pso_multimodal_paper.tex/pdf
│   ├── gpu_pso_multimodal_workshop.tex/pdf
│   └── gpu_pso_multimodal_appendix.tex/pdf
├── .venv/                            # Python virtual environment (pip)
├── .venv311/                         # Python 3.11 virtual environment (pip)
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

---

## Quick Start

### 1. Compile GPU Runners

```bash
cd src/gpu
xcrun swiftc metal_gpu_runner.swift -framework Metal -o ../../build/metal_gpu_runner
xcrun swiftc metal_gpu_runner_highdim.swift -framework Metal -o ../../build/metal_gpu_runner_highdim
cd ../..
```

### 2. Run 2D Benchmarks

```bash
# Plain PSO
./build/metal_gpu_runner --mode plain

# PSO--KMeans (fixed-K)
./build/metal_gpu_runner --mode kmeans

# Enhanced PSO--KMeans (adaptive-K + refinement + policy)
./build/metal_gpu_runner --mode kmeans --enable-refinement --enable-adaptive-k --adaptive-policy
```

### 3. Run High-Dimensional (5D, 10D)

```bash
# 5D Rastrigin with adaptive policy
./build/metal_gpu_runner_highdim --dimension 5 --objective rastrigin --mode kmeans_enhanced --trials 8

# 10D Ackley with fixed-K clustering
./build/metal_gpu_runner_highdim --dimension 10 --objective ackley --mode kmeans --trials 4
```

### 4. Generate Analysis (Python)

```bash
# Activate virtualenv
source .venv311/bin/activate

# Generate plots from GPU outputs
python src/python/plot_gpu_outputs.py

# Run comparative baseline study
python src/python/run_comparative_study.py

# Run robustness sweep (20 trials)
python src/python/run_review_enhancements.py --trials 20

# Validate high-dimensional Metal implementation
python src/python/run_highdim_metal_validation.py
```

---

## Key Algorithms

### PSO Dynamics
Each particle has position $\mathbf{x}_i$, velocity $\mathbf{v}_i$, personal best $\mathbf{p}_i$, and fitness.

$$\mathbf{v}_i^{t+1} = w \mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{a}_i^t - \mathbf{x}_i^t)$$
$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \mathbf{v}_i^{t+1}$$

The attractor $\mathbf{a}_i^t$ is either:
- **Global best** (plain mode): $\mathbf{a}_i^t = \mathbf{g}^t$
- **Cluster best** (niching mode): $\mathbf{a}_i^t = \mathbf{g}_{k(i)}^t$, where $k(i)$ is K-means cluster assignment

### K-means Coupling
Particle-best states are clustered at each iteration:
$$k(i) = \arg\min_{k} \|\mathbf{p}_i - \boldsymbol\mu_k\|_2^2$$
$$\boldsymbol\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{p}_i$$

Weak clusters (low occupancy) are reseeded to far-away particle-best candidates to maintain diversity.

### Adaptive-K Policy
- **Expansion**: If largest cluster $> 0.72 \times N/K$ → increase $K$
- **Shrinkage**: If smallest cluster $< 0.30 \times N/K$ → decrease $K$
- **Objective-aware thresholds**: Per-objective policy overrides (e.g., Ackley: 0.68/0.25)

---

## Command-Line Reference

### 2D Runner (`metal_gpu_runner`)

| Flag | Type | Description |
|------|------|-------------|
| `--mode` | string | `plain`, `kmeans`, or `kmeans_enhanced` |
| `--objective` | string | Objective name or `all` (default: all) |
| `--particles` | int | Swarm size (default: 256; 512 for Hölder) |
| `--iterations` | int | PSO iterations (default: 500) |
| `--seed` | int | Random seed |
| `--enable-adaptive-k` | flag | Enable adaptive-K logic |
| `--enable-refinement` | flag | Post-run local refinement |
| `--adaptive-policy` | flag | Objective-aware policy thresholds |
| `--verbose` | flag | Print per-iteration metrics |

### N-D Runner (`metal_gpu_runner_highdim`)

| Flag | Type | Description |
|------|------|-------------|
| `--dimension` | int | Dimensionality ($\geq 3$; default: 5) |
| `--objective` | string | `rastrigin` or `ackley` |
| `--mode` | string | `plain`, `kmeans`, or `kmeans_enhanced` |
| `--particles` | int | Swarm size (default: 256) |
| `--iterations` | int | PSO iterations (default: 500) |
| `--trials` | int | Independent runs (default: 1) |
| `--output-dir` | string | Results directory (default: `results/paper_results/`) |

---

## Evaluation Benchmarks

### 2D Objectives (5 benchmarks)
- **Rastrigin**: Highly periodic with many local minima
- **Ackley**: Broad outer region, sharp central basin
- **Himmelblau**: 4 well-separated global minima  
- **Six-Hump Camel**: 2 global + several local minima
- **Hölder Table**: Steep, irregular with narrow pockets

### High-Dimensional Objectives (2 benchmarks)
- **Rastrigin-5D / 10D**: Multimodal periodic landscape
- **Ackley-5D / 10D**: Sharp basin with broader outer region

### Metrics
- **Best fitness**: Global best value found
- **Mean centroid fitness**: Average cluster center objective value
- **Mean distance**: Avg. distance from centroids to known minima ($\epsilon = 0.25$)
- **Coverage ratio**: Fraction of known minima within threshold
- **Distinct centroids**: Number of active clusters
- **Largest cluster fraction**: Concentration in dominant niche
- **Cluster entropy**: Dispersion across niches
- **Adaptive-$K$ changes**: Number of cluster count adjustments

---

## Key Results

### 2D Baseline Comparisons (20-trial robustness)
| Objective | Comparison | p-value |
|-----------|-----------|---------|
| Himmelblau | Plain vs. K-means | 0.0002 *** |
| Six-Hump Camel | Plain vs. K-means | 0.0002 *** |
| Hölder Table | Plain vs. K-means | 0.0002 *** |

**Key Findings:**
- K-means niching significantly improves mean-distance and coverage vs. plain PSO
- Enhanced mode (adaptive-$K$) shows objective-dependent gains
- Basin recovery nearly complete on Himmelblau and Six-Hump Camel

### High-Dimensional Metal Validation (8 trials per condition)
- **Rastrigin-10D**: Plain K-means mean centroid $f$ = 108.9 vs. Plain = 181.5 (p=0.0104)
- **Ackley-10D**: Enhanced policy centroid $f$ = 1.99 vs. K-means = 3.17 (p=0.0190)

### Application Surrogate Study (20 trials)
- **Distinct good configs**: K-means 1.5 vs. Plain 1.0
- **Best task utility**: K-means 0.4321 vs. Plain 0.3975
- **Mean top-2 utility**: K-means 0.4304 vs. Plain 0.3975

---

## Performance (Apple M2 GPU)

### Single-Objective Runtime (500 iterations, 256 particles)
| Objective | Plain | K-means | Enhanced |
|-----------|-------|---------|----------|
| Himmelblau (2D) | 0.11 s | 0.30 s | 0.37 s |
| Six-Hump Camel (2D) | 0.13 s | 0.35 s | 0.40 s |
| Rastrigin-5D | 0.15 s | 0.32 s | 0.39 s |
| Ackley-10D | 0.18 s | 0.35 s | 0.42 s |

Scaling: approximately linear with swarm size and iteration count (64–1024 particles, 100–1000 iterations).

---

## Documentation

### Primary Deliverables
- **`docs/manuscripts/gpu_pso_multimodal_journal.pdf`**: Comprehensive academic manuscript (19 pages)
  - Full methodology, algorithmic details, extensive results, discussion, significance testing
  - Includes explicit ring/lbest baseline, mechanism ablations, coupling-frequency sweep, and warm-up+median timing protocol (see reviewer-gap subsection and Table `tab:reviewer-gap-study`)
  - Suitable for peer-reviewed publication
  
- **`docs/manuscripts/gpu_pso_multimodal_documentation.pdf`**: Practical user guide (8 pages)
  - Quick start, CLI reference, output format specifications, configuration, troubleshooting
  - Suitable for end-users and developers

### Reviewer-Gap Evidence Artifacts
- `src/python/run_reviewer_gap_study.py`: Reproducible experiment runner for ring/lbest baseline, 3 ablations, and coupling sweep
- `results/paper_results/reviewer_gap_study_trials.csv`: Per-trial outcomes (20-trial protocol)
- `results/paper_results/reviewer_gap_study_summary.csv`: Aggregated metrics used in manuscript narrative/table

### Additional References
- `docs/CONSOLIDATION_SUMMARY.md`: Manuscript consolidation notes
- `docs/references/journal_references.bib`: Complete bibliography

---

## Reproducibility

All scripts are deterministic given a seed. To reproduce results:

```bash
# Compile
cd src/gpu
xcrun swiftc metal_gpu_runner.swift -framework Metal -o ../../build/metal_gpu_runner
xcrun swiftc metal_gpu_runner_highdim.swift -framework Metal -o ../../build/metal_gpu_runner_highdim
cd ../..

# Run 2D benchmarks (20-trial robustness)
./build/metal_gpu_runner --mode plain --seed 42 --iterations 500
./build/metal_gpu_runner --mode kmeans --seed 42 --iterations 500

# Generate plots
source .venv311/bin/activate
python src/python/plot_gpu_outputs.py

# Outputs in: results/paper_results/, results/plots/
```

---

## System Requirements

- **Hardware**: Apple M-series GPU (M1, M2, M3 or later recommended)
- **OS**: macOS 12.0 or later
- **Build tools**: Xcode command-line tools (`xcode-select --install`)
- **Python**: 3.8+ (numpy, matplotlib, pandas, scipy)

---

## File Sizes

| Component | Size | Notes |
|-----------|------|-------|
| Journal manuscript (PDF) | 3.6 MB | 15 pages, all figures embedded |
| Documentation (PDF) | 181 KB | 7 pages, practical reference |
| High-D validation results | ~2 MB | CSVs + metadata |
| Plots (PNG) | ~50 MB | 5 objectives + 6 result visuals |
| Temporary files | ~200 MB | Raw trial outputs (cleanup recommended) |

---

## Limitations and Future Work

### Current Limitations

**Scope & Interpretation:**
- Evaluation is primarily strongest on **2D synthetic benchmarks** with well-defined basins and known global optima. Performance on real-world problems with unknown/uncountable optima requires adapted coverage metrics (e.g., clustering coefficient, quality indicators).
- **High-dimensional (d > 10) scaling** remains pilot-stage. Native Metal N-D kernels are functional at d=5 and d=10, but not yet optimized for d > 20. Production deployment at d ≥ 50 would require kernel architecture revisiting (buffer management, memory access patterns).
- **Adaptive-$K$ and policy thresholds** are tuned heuristically based on 2D/low-D observations. Transfer to new problem classes (e.g., constrained, discrete, dynamic) is not guaranteed without re-tuning.

**Hardware & Software:**
- **Apple Silicon exclusive**: No cross-platform support (CUDA, SYCL, OpenCL, Vulkan). Reproducibility limited to M-series MacBooks.
- **Fixed particle & cluster counts**: Current pipeline assumes 256–1024 particles and K=4 nominal. Automatic scaling strategies (dynamic population sizing, cluster count adaptation beyond simple occupancy rules) are not implemented.
- **Limited refinement strategies**: Local centroid refinement uses hill-climbing. More sophisticated local search (e.g., Nelder–Mead, Powell's method) could improve basin tightness but adds computational overhead.

**Evaluation Design:**
- **Metrics focus on centroid proximity** rather than basin volume or basin quality diversity. Functions with irregular basin shapes (e.g., isolated narrow pockets) may report false negatives (centroid near optimum but not "counted" within $\epsilon$-threshold).
- **Single-seed robustness**: Although 20-trial studies are reported, larger multi-seed campaigns (100+ trials per objective) would strengthen statistical confidence, especially on high-variance objectives.
- **No real-world application tuning**: Application surrogate study uses synthetic task utility function. Real engineering/ML applications would need domain-specific objective engineering and constraint handling.

### Future Work (Priority Order)

**Priority 1: Cross-Backend Portability (Major Impact)**
- Implement CUDA backend for NVIDIA GPUs and cloud deployment (GCP, AWS, Azure)
- Port Metal kernels to SYCL/oneAPI for Intel and broader hardware support
- Create automated backend selection (detect GPU type, compile appropriate kernels)
- **Benefit**: Enables broader reproducibility, cloud scaling, real-world benchmark access (e.g., HPCC benchmarks, ML pipeline tuning)

**Priority 2: Real-World Objective Suites (Validation Challenge)**
- Integrate engineering design optimization benchmark suite (e.g., MOPTA, AutoML benchmark, structural design)
- Add constraint handling layer (linear/nonlinear inequality and equality constraints)
- Implement penalty and barrier methods for soft/hard constraint satisfaction
- **Benefit**: Demonstrates practical utility beyond toy benchmarks; informs algorithm parameter sensitivity to real problem structure

**Priority 3: Scalability & Optimization (Performance)**
- Profile and optimize Metal kernel memory access patterns (SoA vs. AoS, cache locality for N-D)
- Implement dynamic population and cluster count auto-scaling based on fitness landscape diagnostics
- Add GPU memory management for swarms > 10,000 particles
- Benchmark vs. multi-threaded CPU baseline to quantify GPU overhead amortization
- **Benefit**: Enables larger-scale multimodal screening in practical timescales

**Priority 4: Advanced Niching Strategies (Algorithmic Depth)**
- Implement clearing and sharing-based niching as alternative to K-means coupling
- Add speciation/ species-distance metrics for more sophisticated sub-swarm management
- Experiment with hybrid approaches (K-means + local topology for finer-grained control)
- **Benefit**: Improves algorithm flexibility for diverse landscape structures

**Priority 5: Theoretical Analysis & Convergence (Rigor)**
- Develop convergence conditions for PSO—KMeans on multimodal landscapes (sufficient conditions for basin discovery)
- Analyze reseeding strategy coverage guarantees and occupancy stability
- Study adaptive-$K$ policy stability and sensitivity to threshold hyperparameters
- **Benefit**: Moves beyond empirical tuning to principled algorithm design

### Realistic Constraints

**Computational Budget:**
- Cross-backend portability requires ~2–3 months of engineering (CUDA, SYCL, Vulkan porting, CI/CD setup)
- Real-world benchmark integration (constraint fitting, domain metric encoding) requires ~1 month per application family
- Theoretical convergence analysis requires domain expertise; outsourcing or extended research timeline needed

**Hardware Access:**
- CUDA testing requires NVIDIA GPU (local or cloud access)
- SYCL/oneAPI testing requires Intel GPU or emulator (less accessible)
- Recommenderion: Focus on CUDA as first cross-backend target (largest installed base)

**Reproducibility Trade-off:**
- Expanding to multiple backends improves reproducibility but increases maintenance burden (keep kernels in sync across platforms)
- Recommend clear kernel versioning and regression testing pipeline

---

## License

MIT License. See individual source files for license headers.

---

## Citation

If you use this code or Results in your research, please cite:

```bibtex
@article{Kolluru2026PSO,
  title={PSO--KMeans Niching for Multimodal Optimization with Metal-GPU Realization on Apple M-Series},
  author={Kolluru, Ramesh},
  year={2026}
}
```

---

## Contact

Project: PSO--KMeans GPU Niching on Apple Metal  
Status: ✓ Complete with consolidated documentation  
Last Updated: February 22, 2026
