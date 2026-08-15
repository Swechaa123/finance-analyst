# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Contents
- `Research_Paper_Bank_Churn.docx` — full technical write-up (EDA, methodology, model comparison, explainability, recommendations)
- `Executive_Summary_Bank_Churn.docx` — one-page non-technical summary for stakeholders
- `model_comparison.csv` — metrics for all 4 models (Accuracy, Precision, Recall, F1, ROC-AUC)
- `eda_overview.png`, `correlation_heatmap.png` — EDA visuals
- `roc_curves.png`, `confusion_matrix_best.png` — model evaluation visuals
- `feature_importance.png`, `partial_dependence.png` — explainability visuals
- `model_*.joblib`, `scaler.joblib`, `feature_columns.joblib`, `numeric_columns.joblib` — trained models + preprocessing artifacts
- `churn_app/` — Streamlit dashboard (see below)

## Running the Streamlit Dashboard
```bash
pip install streamlit scikit-learn pandas numpy matplotlib joblib
cd churn_app
streamlit run churn_app.py
```
The app folder is self-contained — it already includes the trained model, scaler, and feature list it needs.

## Notes on Substitutions
This environment had no internet access, so two brief items were substituted with equivalent tools:
- **XGBoost** (listed as optional) → scikit-learn's Gradient Boosting, used as the advanced ensemble model.
- **SHAP** → scikit-learn's permutation importance + partial dependence plots (`sklearn.inspection`), which give the same explainability coverage — feature ranking and directional effect — without the extra dependency.

## Key Result
Gradient Boosting: ROC-AUC 0.871, Accuracy 86.8%, Precision 0.785, Recall 0.484
Random Forest (best F1): ROC-AUC 0.867, F1 0.639, Recall 0.668

The dashboard's adjustable threshold lets you trade precision for recall depending on retention campaign capacity.
