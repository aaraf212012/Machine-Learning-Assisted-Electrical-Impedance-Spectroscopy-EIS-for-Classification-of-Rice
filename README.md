# Machine Learning Assisted Electrical Impedance Spectroscopy (EIS) for Classification of Rice

This repository provides the official multimodal dataset and reproducible Python evaluation pipelines for the paper:
> **"Machine Learning Assisted Electrical Impedance Spectroscopy (EIS) for Classification of Rice"**  
> *IEEE Access*, 2026.

---

## 1. Repository Structure

```
github_repository/
│
├── Dataset.csv              # Full 500-sample multimodal dataset (100 vendors, 5 varieties)
├── evaluate_models.py       # Python evaluation script (Repeated Stratified CV, GroupKFold, Ablation)
├── requirements.txt         # Required Python packages
└── README.md                # Documentation and reproduction guide
```

---

## 2. Dataset Demographics & Features

The dataset comprises **500 independent commercial rice samples** sourced from **100 distinct retail vendors** across four major market divisions in Dhaka, Bangladesh (Notunbazar, Rampura, Demra, and Mirpur). Each vendor supplied one 250 g retail lot of five commercial rice varieties:
1. **BRRI Dhan 28** (Class 1, 100 samples)
2. **Miniket** (Class 2, 100 samples)
3. **Paijam** (Class 3, 100 samples)
4. **Nazir** (Class 4, 100 samples)
5. **Chinigura** (Class 5, 100 samples)

### Feature Descriptions (17 Candidates):
- **EIS Impedance Magnitudes ($|Z|$):** `F1_Impedance` to `F7_Impedance` (5 kHz to 35 kHz in 5 kHz steps, $\Omega$).
- **EIS Phase Angles ($\theta$):** `F1_Phase` to `F7_Phase` (5 kHz to 35 kHz in 5 kHz steps, degrees).
- **Physical Morphological Metrics:**
  - `L/W Ratio`: Grain length-to-width ratio measured via digital caliper.
  - `Bulk Weight`: Mass of grain in standardized fixed-volume container (g).
  - `1000_Grain_Weight`: Mass of 1000 grains (g).
- **Target Label:** `Label` (integer values 1 through 5).

---

## 3. Installation & Requirements

Ensure you have Python 3.8+ installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`):
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

---

## 4. Reproducing Paper Results

Execute the main evaluation script to reproduce all reported benchmarks:

```bash
python evaluate_models.py
```

### What this script outputs:
1. **10-Fold Repeated Stratified Cross-Validation (5 Repeats, 50 evaluations):**
   - SVM: $90.32\% \pm 4.62\%$ (Weighted F1: 0.903)
   - Random Forest: $88.88\% \pm 4.42\%$ (Weighted F1: 0.889)
   - Multi-Layer Perceptron (MLP): $87.92\% \pm 4.65\%$ (Weighted F1: 0.879)
2. **10-Fold Vendor-Level GroupKFold Validation:**
   - Evaluates generalization across independent retail lots without vendor-level data leakage.
   - Generates the pooled $5 \times 5$ confusion matrix (Fig. 7) and per-variety classification report (Table 5).
3. **Ablation Study Across Feature Subsets (Table 4):**
   - 2-feature morphology baseline (`L/W Ratio` + `1000_Grain_Weight`)
   - 3-feature morphology baseline (`L/W Ratio` + `Bulk Weight` + `1000_Grain_Weight`)
   - 14-feature EIS alone (magnitudes + phase angles)
   - 4-feature rapid screening tier (without grain counting)
   - 4-feature selected multimodal model (RFECV)
   - 17-feature full candidate set
4. **Statistical Significance Tests:**
   - Paired two-tailed $t$-tests and Nadeau–Bengio resampled corrected $t$-tests.

---

## 5. Citation

If you use this dataset or code in your research, please cite:
```bibtex
@article{araf2026rice_eis,
  title={Machine Learning Assisted Electrical Impedance Spectroscopy (EIS) for Classification of Rice},
  author={Araf, Al and Shovon, Md. Rafiul Islam and Rafi, Abdullah Al and Saha, Arpita and Akter, Farhana and Hossain, Md. Rabiul and Islam, Md. Kafiul and Shahnaz, Celia},
  journal={IEEE Access},
  volume={14},
  year={2026},
  doi={10.1109/ACCESS.2026.0429000}
}
```

---

## 6. License
This project and dataset are distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
