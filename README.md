# Advanced Evaluation and Benchmarking of MACE (Machine Learning Interatomic Potentials) via ASE

This repository contains a rigorous analytical pipeline formulated in Google Colab notebooks. It is dedicated to the instantiation, thermodynamic benchmarking, and comparative topological analysis of **MACE (Machine Learning Interatomic Potentials)**. MACE leverages higher-order E(3)-equivariant message-passing neural networks (MPNNs) to construct surrogate models that exhibit ab initio (Density Functional Theory - DFT) fidelity with highly scalable linear-scaling $O(N)$ computational complexity. 

The evaluation framework is deeply integrated with the **Atomic Simulation Environment (ASE)**, permitting extensive molecular dynamics (MD) phase-space exploration, deterministic and stochastic integration schemes, and the derivation of thermodynamic observables.

## 📓 Notebook Architectures and Topological Implementations

### 1. `MACE choice.ipynb`
This notebook implements a multiplexed architectural comparison protocol. It systematically instantiates various MACE pre-trained checkpoints (including `MACE-MP-0a`, `MACE-MP-0b3`, `MACE-MPA-0`, `MACE-OMAT-0`, and specific exchange-correlation functional regressors like `MACE-MATPES-PBE-0` and `MACE-MATPES-r2SCAN-0`).
* **Analytical Scope:** It maps the many-body potential energy surfaces (PES) and computes gradients (atomic forces) and virial stress tensors to evaluate the models' zero-shot inferential capacity and out-of-distribution (OOD) generalization across diverse stoichiometric configurations.
* **Integrators:** Implements both Microcanonical (NVE) via Symplectic Velocity Verlet integrators and Canonical (NVT) ensembles via Stochastic Langevin thermostats.

### 2. `mace-mp-0a-small.ipynb`
A focused analytical pipeline examining the low-parameter regime (`small` variant) of the `mace-mp-0a` foundational model.
* **Objective:** Assesses the inferential latency and heuristic accuracy tradeoffs. Focuses on phase-space trajectory stability under prolonged Langevin dynamics, monitoring the Hamiltonian phase-space conservation $H(q,p) = T(p) + V(q)$, and instantaneous temperature fluctuations relative to the Maxwell-Boltzmann target distribution.

### 3. `mace-mp-0a-medium.ipynb`
Deploys the intermediate parametric configuration (`medium` variant) for optimized force field evaluation. 
* **Objective:** Acts as the primary benchmark for investigating thermal equilibration. Employs `ase.neighborlist` to construct the local contiguous topological environment and `ase.geometry.analysis.Analysis` to extract radial distribution functions (RDF) and polyhedral coordination symmetries over the generated MD trajectories.

### 4. `mace-mp-0a-large.ipynb`
Designed for maximal representational capacity using the highly-parameterized `large` variant of the `mace-mp-0a` checkpoint.
* **Objective:** Prioritizes absolute energetic and force-vector fidelity. This notebook is tailored for demanding configurational spaces where higher-order multi-body spherical harmonic expansions (high $L$-max) and extended tensor product representations are necessary to resolve complex non-covalent or highly correlated electronic effects encoded within the geometric embeddings.

### 5. `Mace-testing.ipynb`
A data-ingestion and preprocessing orchestrator. 
* **Objective:** Facilitates the bridging of the Colab runtime environment with persistent Google Drive storage arrays. Specifically designed to handle the localized extraction and hierarchical parsing of large-scale ab initio trajectory datasets (e.g., the MPtrj dataset archive), preparing the atomic structures for subsequent topological featurization and model inference.

## ⚛️ Thermodynamic Integrators & Kinetic Protocols

Across the evaluation suites, the following rigorous MD protocols are enforced:
* **Velocity Initialization:** Atomic momenta are initialized utilizing `ase.md.velocitydistribution.MaxwellBoltzmannDistribution` and `Stationary` to ensure strict adherence to the target equipartition theorem while nullifying any net translational center-of-mass momentum.
* **Langevin Thermostatting:** Coupling to a fictive stochastic heat bath via `ase.md.langevin.Langevin`, employing specific scalar friction coefficients ($\gamma$) to govern the characteristic thermalization decay rate and govern the NVT ensemble phase-space density.
* **Symplectic Integration:** Implementation of `ase.md.verlet.VelocityVerlet` to ensure time-reversal symmetry and strict phase-space volume conservation (Liouville's theorem) for unthermostatted (NVE) trajectory segments.

## 🛠️ Compute Environment and Tensor Backend Dependencies

The local execution environment mandates specific hardware acceleration and tensor compilation modules to execute the underlying E(3)-equivariant operations (such as Clebsch-Gordan tensor products and spherical harmonic projections) efficiently:

* `mace-torch`: The core architectural repository for Equivariant Interatomic Potentials.
* `ase` (Atomic Simulation Environment): Required for building the atomistic graph structures, calculators, and MD integrators.
* `cuequivariance-torch`: A highly optimized backend providing CUDA-accelerated kernels for evaluating equivariant tensor operations, significantly reducing inferential bottlenecking.

**Local Environment Setup:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mace-torch ase cuequivariance-torch scipy matplotlib tqdm
```

## 🚀 Execution Directives

1. **Colab Instantiation**: Initialize the specific `.ipynb` architecture within a Google Colab instance.
2. **Hardware Acceleration**: To mitigate prohibitive evaluation latencies associated with high-order tensor products, binding the runtime to a CUDA-enabled GPU (T4 Tensor Core or advanced Ampere architecture) is strictly mandatory (`Runtime` -> `Change runtime type` -> `Hardware accelerator` -> `GPU`).
3. **Sequential Execution**: Execute the topological mapping, calculator instantiation, and MD loop routines sequentially. For notebooks requiring structural priors (`Mace-testing.ipynb`), ensure the requisite Cartesian coordinate archives (e.g., `MPtrj.zip`) are adequately mounted to the `/content/drive` subsystem.
