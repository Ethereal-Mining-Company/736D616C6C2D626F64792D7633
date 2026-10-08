# Orbital Sizing & Assessment Protocol (OSAP)
*Predicting Asteroid Diameter Based on Photometric Parameters and Orbital Characteristics for Mining of Precious Metals for Use in MHD Satellites* \
Commissioned by the [Ethereal Mining Company (EMC)](https://github.com/Ethereal-Mining-Company)

![Solar System Orbital Sorting Map](images/asteroid_orbital_sorting_map.png)

>[!CAUTION]
>This is a fictional project as part of a broader D&D universe campaign for entertainment use only. It may be integrated into educational projects as flavor text to help simulate working in real-world environments.

>[!NOTE]
>This `README.md` is flavor text for entertainment use only to be used with an educational project. The project data is from the [JPL NASA Small-Body Dataset](https://ssd.jpl.nasa.gov/tools/sbdb_query.html) that is being analyzed for a graded educational project.

## EXECUTIVE SUMMARY
**Project Title:** Orbital Sizing & Assessment Protocol (OSAP) \
**Description:** Predicting Asteroid Diameter Based on Photometric Parameters and Orbital Characteristics for Mining of Precious Metals for Use in MHD Satellites \
**Prepared For:** Almeric Vein-Wright, Vice President of Mining Operations, [Ethereal Mining Company (EMC)](https://github.com/Ethereal-Mining-Company) \
**Target Application:** Pre-Sourcing Feasibility Assessment for Magnetohydrodynamic (MHD) Satellite Production

### Enterprise Directive & Operational Mandate
> **[AI DIRECT QUOTE]** \
>The **[Ethereal Mining Company (EMC)](https://github.com/Ethereal-Mining-Company)** has authorized a strategic expansion mandate requiring the rapid deployment of Magnetohydrodynamic (MHD) satellites. These orbital units are engineered to construct massive inductive siphons, capturing low-density ambient electromagnetic waves and artificially pressurizing them into high-energy streams. This localized cosmic pipeline serves a singular corporate objective: fueling the infrastructure required to expand EMC's industrial capacity and territorial reach across the deep galaxy.
>
>Executing this grand architecture demands vast quantities of precious heavy metals, which must be extracted directly from Small Solar System Bodies. However, to ensure physical extraction is both structurally feasible and profitable for the counting-house, target bodies must exceed strict baseline dimensions. Furthermore, corporate dispatch protocols dictate that the exact volumetric measurements of these objects must be mapped precisely beforehand; knowing the definite dimensions allows the executive board to accurately size, equip, and budget physical extraction teams, preventing costly under-allocations or wasteful over-deployments of heavy off-world machinery.



## Asteroid Diameter Prediction Pipeline

[Python 3.11+](https://python.org)
[Keras 3](https://keras.io)
[Scikit-Learn](https://scikit-learn.org)
[Git LFS](https://git-lfs.com)

An end-to-end machine learning engineering framework built to estimate asteroid diameters utilizing data points extracted from the [JPL NASA Small-Body Dataset](https://ssd.jpl.nasa.gov/tools/sbdb_query.html). 

This project implements a multi-model training matrix comparing **Traditional Gradient-Boosted Tree Ensembles** against **Deep Learning Tabular Architectures** (Tabular ResNet, Feature Tokenizer Transformer). The system evaluates performance changes when absolute magnitude (H) is removed from the input arrays to determine if orbital characteristics alone contain sufficient predictive signal for diameter approximation.



## Project Architecture
```text
├── README.md                 # Project documentation and setup instructions
├── config.yaml               # Unified pipeline configuration and hyperparameter bounds
├── asteroid.py               # Core execution, cross-validation router, and metrics logging engine
├── visualization_utils.py    # Custom presentation-grade matplotlib/seaborn visualization modules
├── main.ipynb                # Master project runner notebook
├── requirements.txt          # Production-locked module dependencies
├── data/                     # Local raw and preprocessed data arrays
├── figures/                  # Performance evaluation charts
├── images/                   # Asset images used in documentation or UI
├── logs/                     # Tracked model engine, fold-by-fold, and predictions results
├── presentations/            # Slide decks and project presentation files
├── references/               # Explanatory materials, papers, or manual references
└── artifacts/               
    └── production/          # Main production deployment assets tracked natively via Git LFS
```



## Evaluation Matrix Leaderboard

All six architectures were evaluated using a **5-Fold Cross-Validation** strategy across **139,203 clean asteroid samples**. 

Performance metrics track both the **Full Feature Space** (including absolute magnitude H) and the **Restricted Feature Space** (purely orbital parameters (a, e, i)), with true feature configurations preserved as stringified arrays inside our tracking ledgers.

| Rank | Model Architecture | Full Model Mean R² | Restricted Model Mean R² | Drop Impact (Δ R²) | Full Model MAE (km) | Restricted Model MAE (km) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **XGBoost Base** | **0.8548** | 0.3630 | 0.4917 | **1.0360** | 2.3461 |
| **2** | **Random Forest Base** | **0.8508** | 0.3330 | 0.5178 | 1.0443 | 2.4090 |
| **3** | **Keras FT-Transformer** | **0.8280** | **0.3104** | 0.5177 | 1.1435 | 2.4215 |
| **4** | **Keras Tabular ResNet** | **0.8189** | 0.2487 | 0.5702 | 1.2734 | 8.3245 |
| **5** | **Keras Wide & Deep** | **0.8053** | 0.2090 | 0.5963 | 1.2608 | 4.10 × 10²² |
| **6** | **Gated ResNet** | **0.7887** | 0.1492 | 0.6396 | 4.4961 | *Exploded (≈ 10²¹)* |



## Hypothesis Testing Framework

The logging system automates an out-of-sample statistical and operational hypothesis reconciliation check to guide target model deployments.
Operational Drop Threshold: ($\delta$) = 0.0500

### 1. Statistical Hypothesis Verification (H₀\_stat)
*   **Hypothesis:** Group 2 features (Absolute Magnitude H) do not significantly improve out-of-sample diameter predictions.
*   **Outcome:** **REJECTED (H₁\_stat)**. Across all evaluations, the performance expansion generated by including H is highly statistically significant.

### 2. Operational Sufficiency Verification (H₀\_ops)
*   **Hypothesis:** The cross-validation performance drop (Δ R²) when moving from the Full feature space to the Restricted feature space is ≤ δ (0.05). The Restricted subset is operationally sufficient for production.
*   **Outcome:** **FAILED TO REJECT (H₀\_ops)**. The actual drop impact ranges from **0.4917 to 0.6396**, which massively violates the operational tolerance boundary.

**Conclusion: The restricted orbital feature subset is operationally insufficient for deployment.**



##  Reproduction & Verification Guide

### 1. Environment Setup
Install the production-locked dependencies directly from your terminal:
```bash
pip install -r requirements.txt
```

### 2. Large File Storage Setup
Ensure Git LFS is properly hooked into your local system environment before interacting with the main deployment binaries:
```bash
git lfs install
git lfs pull
```

### 3. Execution Sweep
Open your notebook environment, navigate to `main.ipynb`, and run the master training block to automatically clear old logs, initialize the six-architecture matrix cross-validation loops, and update your local performance databases.
```bash
jupyter notebook main.ipynb
```
