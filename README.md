# Resonant Mode Detectability in Binary Neutron Star Waveforms

Software developed to study the detectability thresholds of resonantly excited oscillation modes in binary neutron star (BNS) gravitational-wave signals.

## Installation

Create the conda environment from the provided requirements file:

```bash
conda create -n <my-env> python=3.12.3
conda activate <my-env>
```

To install the required packages from the `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

## Repository Structure

### `detectability/`

Main package containing the core classes and functions used throughout the project, including:

* Binary neutron star inspiral and waveform calculations.
* Resonant mode modeling.
* Match-filtering and detectability analysis tools.
* Utilities for computing signal overlaps and detection thresholds.

### `applications/`

Collection of Jupyter notebooks used to reproduce the figures and results presented in the associated paper.

## Purpose

The code evaluates the imprint of neutron-star oscillation modes on gravitational-wave signals and determines the conditions under which these resonances become detectable with current and future gravitational-wave detectors.
