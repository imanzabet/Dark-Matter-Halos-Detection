# Dark Matter Halos Detection using Constrained Multivariate Gaussian Mixture Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![University of Vienna](https://img.shields.io/badge/University%20of%20Vienna-Faculty%20of%20Physics-006699.svg)](https://www.univie.ac.at/)

An unsupervised statistical modeling and machine learning framework designed to detect, characterize, and segment **Dark Matter (DM) Halos and Subhalos** from 3D cosmological N-body simulation data (Bolshoi Simulation).

This project integrates **3D Kernel Density Estimation (KDE) local maxima peak-finding** with a **physically constrained finite mixture model (Einasto Profiles + Uniform Background)** optimized via Maximum Likelihood Estimation (MLE).

---

## 📌 Scientific & Computational Motivation

### 1. The Astrophysical Challenge
* Dark matter halos do not emit electromagnetic radiation; their presence must be inferred through gravitational effects on luminous matter or N-body spatial coordinates.
* Standard spatial point process models (e.g., Neyman Poisson models or de Vaucouleurs log-normal distributions) fail to capture the multi-scale, cuspy nature of overlapping halos and diffuse cosmic backgrounds.
* Halos follow steep radial density distributions that standard spherical Gaussians cannot accurately represent.

### 2. The Data Science Challenge
* Standard unsupervised clustering algorithms (**K-Means, DBSCAN, Agglomerative, Spectral Clustering**) perform poorly on continuous, highly skewed, non-uniform spatial densities and fail to assign probabilistic memberships in dense regimes.
* Non-convex likelihood landscapes in high dimensions make standard EM/MLE susceptible to getting trapped in local extrema without principled parameter initialization.

---

## 🔬 Methodology & Pipeline Architecture

```
 ┌──────────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
 │  3D Simulation Data  │ ──► │  3D Grid KDE Seeding  │ ──► │ Constrained MLE / GMM  │
 │  (Bolshoi N-body)    │     │  (26-Voxel Maxima)    │     │  (Einasto + Uniform)   │
 └──────────────────────┘     └───────────────────────┘     └───────────┬────────────┘
                                                                        │
 ┌──────────────────────┐     ┌───────────────────────┐                 │
 │ Residual Analysis    │ ◄── │ Posterior Soft/Hard   │ ◄───────────────┘
 │ (AIC, BIC, Pearson)  │     │ Cluster Assignment    │
 └──────────────────────┘     └───────────────────────┘
```

### 1. Exploratory Data Analysis & Baselines
* **Data Source:** Bolshoi Cosmological Simulation box ($h^{-1}\text{Mpc}$ coordinates).
* **Elbow Analysis:** Within-Cluster Sum of Squares (WCSS) via `kneed` to determine candidate component counts ($K \in \{3, 7, 9\}$).
* **Baseline Clustering:** Tested K-Means, DBSCAN, Agglomerative, and Spectral Clustering to highlight boundary and density limitations.

### 2. Parameter Seeding via 3D Kernel Density Estimation (KDE)
To prevent convergence to suboptimal local maxima, initial candidate centers are proposed using an isotropic 3D Gaussian KDE grid:
$$\hat{f}(\mathbf{x}) = \frac{1}{n h^3} \sum_{i=1}^n K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$
* Evaluated over a discrete 3D spatial grid (e.g., $55 \times 55 \times 27$).
* Local maxima are extracted by evaluating each voxel against its **26 neighboring cells in 3D space** and sorting peaks by density magnitude.

### 3. Constrained Mixture Model (Einasto Profile)
The point distribution is modeled as a finite mixture of $K$ physical halos plus a uniform background component:
$$p(\mathbf{x} \mid \Theta) = \pi_{\text{bg}} f_{\text{bg}}(\mathbf{x}) + \sum_{k=1}^K \pi_k f_k(\mathbf{x} \mid \boldsymbol{\theta}_k)$$

Each halo component follows a 3D **Einasto density profile**:
$$\rho_E(r) \propto \exp\left( -d_n \left[ \left(\frac{r}{r_s}\right)^{1/n} - 1 \right] \right), \quad r = \|\mathbf{x} - \mathbf{x}_0\|$$
* $\mathbf{x}_0 = (x_0, y_0, z_0)$: Halo centroid.
* $r_s > 0$: Scale radius.
* $n > 0$: Einasto shape parameter (curvature index).
* $d_n \approx 3n - \frac{1}{3} + \frac{0.0079}{n}$: Normalization parameter.

### 4. Constrained Optimization
The Negative Log-Likelihood (NLL) is minimized under physical constraints:
* $\sum_{k} \pi_k = 1, \quad \pi_k \ge 0$ (enforced via Softmax parameterization)
* $r_{s,k} > 0, \quad n_k \in (0, 30)$ (log-space / parameter bounds)
* Benchmark of optimization solvers: **Nelder-Mead, BFGS, L-BFGS-B, SLSQP, and trust-constr**. `SLSQP` and `L-BFGS-B` demonstrate superior convergence and boundary enforcement.

### 5. Posterior Membership & Model Diagnostics
* **Soft Assignment:** Posterior probability $\gamma_{i,k} = P(\theta_k \mid \mathbf{x}_i)$ computed using Bayes' theorem.
* **Hard Assignment:** $\hat{k}(\mathbf{x}_i) = \arg\max_k \pi_k f_k(\mathbf{x}_i \mid \boldsymbol{\theta}_k)$.
* **Model Selection:** Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
* **Goodness-of-Fit:** 3D Raw ($O - P$) and Pearson Residuals $\left(\frac{O - P}{\sqrt{P}}\right)$ evaluated across 2D slices along the $z$-axis.

---

## 📊 Summary of Results

| Metric / Parameter | Value / Finding |
| :--- | :--- |
| **Analyzed Volume Box** | $X \in [18, 43], Y \in [171, 196], Z \in [95, 102]\; h^{-1}\text{Mpc}$ |
| **Optimal Components** | $K = 7$ Halos + 1 Uniform Background |
| **Best Optimizer** | Sequential Least Squares Programming (**SLSQP**) |
| **Final Negative Log-Likelihood** | $\approx 4348.70$ |
| **Information Criteria** | $\text{AIC} = 8783.41 \quad\mid\quad \text{BIC} = 9025.95$ |
| **Residual Check** | Centered around zero with negligible spatial systematic bias |

---

## 📁 Repository Structure

```text
├── data/
│   └── datacat.txt               # Bolshoi simulation catalog coordinates
├── notebooks/
│   ├── Dark_Matter_Halos.ipynb   # Main Jupyter/Colab analysis notebook
│   └── Project_Utility.ipynb     # Helper classes (Point3DProcess, quad_3d)
├── src/
│   ├── o1_dark_full_trans.py     # Einasto likelihood, KDE peak finder, solvers
│   ├── models.py                 # Mixture model & density profile definitions
│   └── visualization.py          # 3D projection and residual slice plots
├── figures/                      # Exported plots (Elbow, 3D KDE, residuals)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9 or higher

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/dark-matter-halo-detection.git
   cd dark-matter-halo-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
Launch the interactive Jupyter notebook:
```bash
jupyter lab notebooks/Dark_Matter_Halos.ipynb
```
Or run the headless pipeline directly:
```bash
python src/o1_dark_full_trans.py --clusters 7 --optimizer SLSQP
```

---

## 📦 Key Dependencies
* **Core Computing:** `numpy`, `scipy` (`scipy.optimize`, `scipy.ndimage`, `scipy.stats`)
* **Data Handling:** `pandas`
* **Clustering & ML:** `scikit-learn`, `kneed`
* **Plotting & 3D Visualization:** `matplotlib` (`mpl_toolkits.mplot3d`)

---

## 📚 References
1. **Hurtado-Gil, L., Kuhn, M. A., Arnalte-Mur, P., Feigelson, E. D., & Martínez, V. (2022).** *DarkMix: Mixture Models for the Detection and Characterization of Dark Matter Halos.* The Astrophysical Journal, 939(1), 34.
2. **Kuhn, M. A., & Feigelson, E. D. (2017).** *Mixture models in astronomy.* arXiv preprint [arXiv:1711.11101](https://arxiv.org/abs/1711.11101).
3. **Everitt, B. S., Landau, S., & Leese, M. (2011).** *Cluster Analysis.* 5th Edition, Wiley, London.

---

## 👤 Author & Academic Context
* **Author:** Iman Zabett
* **Course:** *Statistical Methods of Data Analysis* (2024W)
* **Institution:** Faculty of Physics, **University of Vienna** (*Universität Wien*)
* **Instructor:** Prof. Waltenberger
