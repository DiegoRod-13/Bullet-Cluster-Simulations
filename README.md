# Bullet Cluster Simulations

This repository contains code from my Summer 2026 research project on idealized simulations of the Bullet Cluster.

The repository includes code for:

* generating spherical and triaxial galaxy-cluster initial conditions;
* combining the main and bullet clusters;
* reading and analyzing output simulation snapshots;
* producing density profiles, dark matter contours, and mock Chandra X-ray maps;
* measuring dark matter peaks and comparing different simulation parameters.

## Structure

```text
bulletsim/            Reusable analysis functions
initial_conditions/   Initial-condition generation
figures/               Analysis and figure scripts
```

The scripts contain configuration blocks where file paths, snapshots, and simulation cases can be changed directly.

Large simulation snapshots and FITS files are not included.

## Installation

```bash
pip install -r requirements.txt
```

## Author

Diego Rodríguez
Ontario Tech University
