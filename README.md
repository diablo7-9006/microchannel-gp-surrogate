# Gaussian Process Surrogate Model for Microchannel Heat Transfer

A machine learning approach to predict local Nusselt number profiles in turbulent microchannel flows, trained on CFD simulation data.

## Overview

This project builds a Gaussian Process (GP) surrogate model that predicts local heat transfer (Nusselt number) along the length of rectangular microchannels. The model is trained on 628 data points from 40 CFD simulations covering three working fluids.

**Key Result:** The GP surrogate achieves **3.5% prediction error** on unseen geometries, outperforming the published correlation (5.8% error) by **40%**.

This work extends the CFD study and correlations published in:

> Rahaman, M. M., Ramachandran, H., Muhamed, H., Dhanalakota, P., A R, A., Das, S. K., and Pattamatta, D. A. (2026). "Turbulent Flow in a Rectangular Microchannel: Development and Validation of Correlations for Nusselt Number and Friction Factor." *ASME Journal of Heat and Mass Transfer*. [DOI: 10.1115/1.4070848](https://doi.org/10.1115/1.4070848)

---

## Why This Matters

### The Problem

Designing microchannel heat exchangers (used in electronics cooling, data centers, EVs) requires predicting how heat transfer varies along the channel length. Engineers typically use:

1. **CFD simulations** — Accurate but slow (hours per case)
2. **Empirical correlations** — Fast but limited accuracy, no uncertainty estimates

### The Solution

A trained surrogate model that:
- Predicts in **milliseconds** instead of hours
- Achieves **lower error** than published correlations
- Provides **uncertainty bounds** so you know when to trust the prediction

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

### Run the Model

```bash
python gp_surrogate_model.py
```

This will:
1. Load the training data from `GP_Training_Data.xlsx`
2. Train the Gaussian Process model
3. Evaluate on held-out test cases
4. Generate comparison plots
5. Save results to `results/`

---

## Data Format

The training data is not included in this repository. To use this model, prepare an Excel file (`GP_Training_Data.xlsx`) with a sheet named `GP_Input_Data` containing the following columns:

| Column | Description | Units | Example Range |
|--------|-------------|-------|---------------|
| `Case` | Simulation case ID | - | 1, 2, 3, ... |
| `Liquid` | Working fluid | - | Water, Acetone, FC-72 |
| `Dh_microns` | Hydraulic diameter | μm | 200 – 1000 |
| `alpha` | Aspect ratio (height/width) | - | 0.4 – 2.0 |
| `Re` | Reynolds number | - | 2500 – 12000 |
| `Pr` | Prandtl number | - | 4 – 12 |
| `x_mm` | Axial position | mm | 0.7 – 50 |
| `x_star` | Dimensionless position: x / (Dh × Re × Pr) | - | 0.00001 – 0.002 |
| `Nu` | Local Nusselt number (target variable) | - | 25 – 150 |

Each row represents one spatial location from one CFD simulation. Multiple rows per case (varying `x_mm`) capture the axial Nu profile.

---

## Methodology

### Model Architecture

```
Inputs (5 features)              Gaussian Process             Output
─────────────────────           ──────────────────           ────────
Dh (hydraulic diameter)  ─┐
α  (aspect ratio)        ─┼──►   Matérn Kernel    ─────►     Nu(x)
Re (Reynolds number)     ─┤      (ν = 2.5)                   + uncertainty
Pr (Prandtl number)      ─┤
x* (dimensionless pos)   ─┘
```

### Why Gaussian Process?

| Method | Pros | Cons |
|--------|------|------|
| **Power-law correlation** | Simple, interpretable | Fixed functional form, no uncertainty |
| **Neural Network** | Flexible | Needs lots of data, no built-in uncertainty |
| **Gaussian Process** | Uncertainty quantification, works with small datasets | Scales poorly to huge datasets |

With 628 data points, GP is the right choice — large enough to learn patterns, small enough for GP to handle efficiently.

### Validation Strategy

We use **grouped cross-validation**: entire cases are held out during training. This prevents data leakage — the model can't "cheat" by seeing nearby points from the same simulation.

```
Standard CV (WRONG for this data):
  Train: Case 1 points [1,2,3,5,6,7], Case 2 points [1,2,3,5,6,7]...
  Test:  Case 1 point [4], Case 2 point [4]...
  → Model memorizes each case, doesn't generalize

Grouped CV (CORRECT):
  Train: Cases [1,2,3,4,5,6,7,8...]
  Test:  Cases [9,10,11...] ← completely unseen geometries
  → Model must generalize to new designs
```

---

## Results

### Performance Comparison

| Method | R² | MAPE | Within ±15% |
|--------|-----|------|-------------|
| Published Correlation | 0.946 | 5.8% | 94% |
| GP Surrogate | **0.981** | **3.5%** | **100%** |

### Visualizations

The script generates four plots in the `results/` folder:

1. **Profile Comparisons** — Nu vs x* for each test case, showing GP prediction (with uncertainty bands) vs. correlation vs. CFD data

2. **Parity Plots** — Predicted vs. actual Nu for both methods

3. **Error Distribution** — Histogram showing GP has tighter, more centered errors

4. **New Geometry Prediction** — Prediction for a geometry not in any training case, demonstrating surrogate capability

### Example Output

```
================================================================================
RESULTS ON TEST SET (8 cases, 104 points)
================================================================================

Method                          R²        MAPE
------------------------------------------------
Published Correlation           0.9458    5.8%
GP Surrogate                    0.9807    3.5%

GP outperforms correlation by 40% in terms of MAPE.
```

---

## Published Correlation

The published correlation for local Nusselt number (Eq. 17 in the paper) is:

```
Nu_x / Nu_fd = 3.4996 × (X/Dh)^(−0.0314) × α^(−0.0294) × Re^(−0.1182) × Pr^(0.0393)
```

where `Nu_fd` is the fully-developed Nusselt number from the Gnielinski correlation:

```
Nu_fd = (f/8)(Re − 1000)Pr / [1 + 12.7 × √(f/8) × (Pr^(2/3) − 1)]
f = (0.790 × ln(Re) − 1.64)^(−2)
```

The GP model learns the relationship directly from data without assuming this functional form, allowing it to capture nonlinear behaviors the power-law formulation misses.

---

## File Structure

```
├── README.md                    # This file
├── gp_surrogate_model.py        # Main code
├── requirements.txt             # Dependencies
├── GP_Training_Data.xlsx        # Training data (not included — see Data Format)
└── results/                     # Generated outputs
    ├── profile_comparisons.png
    ├── parity_plots.png
    ├── error_distribution.png
    └── new_geometry_prediction.png
```

---

## Citation

If you use this code or methodology, please cite the associated paper:

```bibtex
@article{rahaman2026turbulent,
  title   = {Turbulent Flow in a Rectangular Microchannel: Development and 
             Validation of Correlations for Nusselt Number and Friction Factor},
  author  = {Rahaman, M. M. and Ramachandran, H. and Muhamed, H. and 
             Dhanalakota, P. and A R, A. and Das, S. K. and Pattamatta, D. A.},
  journal = {ASME Journal of Heat and Mass Transfer},
  year    = {2026},
  doi     = {10.1115/1.4070848}
}
```

---

## License

MIT License — feel free to use and modify for your own projects.

---

## Author

**Harshavardhan Ramachandran**  
MS Mechanical Engineering, Texas A&M University  
[LinkedIn](https://www.linkedin.com/in/harshavardhan--r/) • [Portfolio](https://harshavardhanr-portfolio.netlify.app/)

