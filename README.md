# Cylinder Scattering in C-CST

This repository contains symbolic and numerical tools for studying elastic wave scattering by cylinders using **Consistent Couple-Stress Theory (C-CST)**. Citation indications will be provided soon.

The project includes six scattering problems:

1. SH wave / rigid cylinder  
2. SH wave / cavity  
3. SH wave / elastic cylinder  
4. P wave / rigid cylinder  
5. P wave / cavity  
6. P wave / elastic cylinder  

The notebooks derive Fourier–Bessel series solutions, compute modal coefficients, reconstruct fields, verify boundary conditions, and plot boundary stresses and dynamic stress concentration factors (DSCF).

---

## Repository Structure
```
src/
├── extra_fns.py
├── wave_plotter.py
├── SH_rigid_cylinder.ipynb
├── SH_cavity.ipynb
├── SH_elastic_cylinder.ipynb
├── P_rigid_cylinder.ipynb
├── P_cavity.ipynb
├── P_elastic_cylinder.ipynb
├── cache/
└── figs/
```
---

## Main Files

### `extra_fns.py`

Contains symbolic helper functions used across all notebooks:

* differential operators in cylindrical coordinates,
* rotation and curvature tensors,
* strain and stress computation in C-CST,
* classical-limit utilities.

---

### `wave_plotter.py`

Plotting utilities for generating publication-style figures:

* `plot_scalar_field`: single field visualization,
* `plot_field_grid`: grid of fields with shared colorbar,
* LaTeX-style rendering support.

Includes features like masked cylinders, percentile clipping, and compact layouts.

---

### SH notebooks

* `SH_rigid_cylinder.ipynb`: rigid boundary condition problem.
* `SH_cavity.ipynb`: traction-free boundary problem.
* `SH_elastic_cylinder.ipynb`: elastic inclusion with transmitted field.

They compute:

* displacement fields,
* boundary-condition verification,
* boundary stresses,
* DSCF.

---

### P notebooks

* `P_rigid_cylinder.ipynb`: rigid cylinder using Helmholtz potentials.
* `P_cavity.ipynb`: traction-free cavity.
* `P_elastic_cylinder.ipynb`: elastic inclusion.

They compute:

* scalar and vector potentials,
* displacement fields,
* boundary stresses,
* DSCF.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/thomas-martinod/cylinder-scattering-ccst.git
cd cylinder-scattering-ccst
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy scipy sympy matplotlib jupyter ipykernel
```

Register kernel:

```bash
python -m ipykernel install --user --name ccst-scattering --display-name "Python (CCST Scattering)"
```

---

## Workflow

1. Run symbolic setup
2. Build boundary-condition system
3. Solve for coefficients
4. Adimensionalize
5. Lambdify expressions
6. Compute fields or stresses
7. Generate figures

---

## Notes

* Results are saved in `figs/`
* Cached data is stored in `cache/`
* Some expressions are very large; matrix-based solutions are used instead of explicit formulas

We could parameterize the problem using material constants like $\rho$ and $\lambda$, but their influence is fully captured through the dimensionless parameters (especially $\hat{k}_S$ and $\hat{k}_P$). For this reason, the implementation is expressed in terms of these nondimensional quantities.

```
```
