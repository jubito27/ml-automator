# ML-Automator 🚀

**ML-Automator** is a powerful, low-code machine learning utility library that automates Feature Engineering, Hyperparameter Optimization, and Model Evaluation.


## 🌟 Features

- **Automated Feature Engineering**: Handle Target Encoding, Scaling, and Imputation in one line.
- **Optuna-Powered Optimization**: Pre-configured search spaces for RandomForest, XGBoost, CatBoost, LightGBM, and more.
- **Deep Evaluation**:
    - Multi-model score comparison.
    - Interactive ROC and Calibration curves using Plotly.
    - Automated Learning Curve analysis.
    - Automatic report generation (CSV/Excel/PNG).

## 📂 Project Structure
Your library is organized into three core modules:
1. `feature_engineering.py`: Data preprocessing and importance extraction.
2. `models_optimizer.py`: Optuna-based hyperparameter tuning.
3. `trainer.py`: Model training, cross-validation, and visualization.

## 🚀 Quick Start

### 1. Automation at its Best
```python
from automater.feature_engineering import FeatureEvaluation
from sklearn.ensemble import RandomForestClassifier

evaluator = FeatureEvaluation(X, y)
processed_x, importance_df = evaluator.fit_all_at_once(RandomForestClassifier())