# Morpho-Computational Calculus (MCC)
**Physics-Informed Autonomous Reasoning Core (PARC)**

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## Overview

The **Morpho-Computational Calculus (MCC)** framework and its core engine, **PARC**, provide a state-of-the-art **Real-Time Neural Surrogate** for modeling chaotic fluid dynamics. 

For 200 years, the Navier-Stokes equations have presented an insurmountable mathematical challenge, forcing physicists to rely on computationally expensive $O(N^3)$ numerical integration (via supercomputers) to simulate chaotic turbulence, weather patterns, and hemodynamics. 

**PARC bypasses traditional numerical integration entirely.** By leveraging a Physics-Informed Neural Network (PINN) built on the SIREN (Sinusoidal Representation Network) architecture, PARC maps complex boundary conditions and chaotic physical laws into a continuous 4D neural manifold. Once trained, the neural network acts as a highly optimized surrogate solver, predicting fluid dynamics in $O(1)$ time complexity (milliseconds).

### Why Neural Surrogates?
*Traditional symbolic regression models frequently fit patterns to coordinate clouds that produce severe mathematical contradictions when tested directly against the actual differential equations.* 

Matching known data points is not the same as discovering new exact physics. Acknowledging this fundamental mathematical limit, MCC abandons the pursuit of simple algebraic curve-fitting. Instead, the Neural Network *is* the mathematics. PARC functions as a standalone, ultra-fast computational engine capable of replacing supercomputers in time-critical scenarios.

## Core Capabilities
- **Real-Time Meteorology:** Predict chaotic atmospheric shifts (Hurricane Lorenz Attractors) instantly, without supercomputer latency.
- **Biomedical Fluid Dynamics:** Calculate pulsating shear stress and radial boundary conservation in arterial aneurysms in real-time, enabling instant diagnostic tools on consumer hardware.
- **Topological Turbulence:** Model exact Beltrami flow cancellations (ABC flows) via neural manifolds.

## Installation

You can install the `morpho_calculus` package directly using pip:

```bash
git clone https://github.com/raghav/morpho_calculus.git
cd morpho_calculus
pip install -e .
```

## Usage

### Generating the Neural Manifold (Dataset Engine)
To train the SIREN network and map the topological geometry of the ABC flow, Lorenz Attractor, and Aneurysm shear:

```bash
morpho-engine
```
*This will generate the 4D coordinate dataset and evaluate the physics loss function.*

### Decoding and Validation (Symbolic Decoder)
To analyze the generated coordinate clouds and output the conserved geometric invariants:

```bash
morpho-decoder
```

## Architecture

1. **`dataset_engine.py`**: Contains the `SIREN_RepNet4D` PyTorch model and the `PhysicsRegistry3D`. It uses differential equations as boundary constraints to train the neural manifold.
2. **`symbolic_decoder.py`**: Validates the output of the SIREN network against exact physical invariants (e.g., Beltrami Field cancellations and Phase Space Volume Contractions).

## License
MIT License. See `LICENSE` for more information.
