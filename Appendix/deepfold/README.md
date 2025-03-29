# Benchmarking DeepFold

Welcome to the repository dedicated to run all tests that performed in Deepfold.

## Overview

This repository facilitates benchmarking tests for Deepfold.

### Implementation
DeepFold is implemented using $\mathbb{F}_{p^2}$, with $p = 2^{61} - 1$ as the base field and Blake3 as the hash function. The chosen code rate is $2^{-3}$. To modify the code rate, adjust the `CODE_RATE` constant.

### Modules
  - **DeepFold**: The multi-linear FRI-based polynomial commitment scheme proposed in paper. Find this mainly in the `deepfold/` directory.
  - **Batch Variant of DeepFold**: The Batch evaluation version of DeepFold proposed in paper. Find this in the `batch/` directory.
  - **Other FRI-based Multi-linears**:
    - BaseFold in `basefold/` directory
    - DEEP-FRI in `fri/` directory
    - PolyFRIM in `polyfrim/` directory
    - Virgo in `virgo/` directory
    - Zeromorph in `zeromorph/` directory

- **Utilities**: All the above protocols leverage utilities found in `util/`, which includes implementations for Merkle trees, finite fields, polynomials, and other necessary tools.

## Setup

1. **Install Rust**: Follow the instructions on [Rust Installation](https://www.rust-lang.org/tools/install).
   
2. **Verify Installation**: Post-installation, ensure everything is set up correctly with:
   ```bash
   cargo --version
   rustup --version
   ```

3. **Use the Nightly Toolchain**: 
   ```bash
   rustup default nightly
   ```

## Benchmarking

- **Benchmark All Protocols**:
```bash
cargo bench
```

-- **Benchmark a Specific Protocol**: Choose from `deepfold`, `basefold`, `fri`, `matmult` `polyfrim`, `virgo` or `zeromorph`
```bash
cargo bench -p <scheme>
```

-- **Output Proof Size**: Choose from `deepfold`, `basefold`, `fri`, `matmult`, `polyfrim`, `virgo` or `zeromorph`
```bash
cargo test --release -p <scheme>
```

> **Note**: The most extensive benchmarking point may require approximately 50 GB of RAM.
