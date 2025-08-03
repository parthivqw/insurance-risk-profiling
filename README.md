# Insurance Risk Profiling and Prediction

This repository contains a machine learning solution for profiling and predicting insurance risk. It includes a complete notebook implementation, a detailed markdown write-up, and a downloadable PDF report.

The solution covers the full pipeline from preprocessing and EDA to model training and segmentation, using:

- LightGBM for premium regression (R² = 0.92)
- Random Forest for risk tier classification (90% accuracy)
- KMeans for client segmentation based on a custom risk score

---

## Contents

- `riskPROFILE.ipynb` – Jupyter notebook with full code and analysis
- `Insurance Risk Profiling and Prediction.md` – Long-form markdown write-up
- `Comprehensive_Report_Insurance_Risk_Profiling_and_Prediction.pdf` – Printable PDF report

---

## Key Features

- Data Cleaning and Feature Engineering
- Handling of Date Columns
- Exploratory Data Analysis (EDA)
- Multi-model pipeline (regression, classification, clustering)
- Visualizations and interpretability plots
- Future-ready for Streamlit or dashboard deployment

---

## Requirements

- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  

(Optional: lightgbm if you implement it fully)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
