# 📦 Automac: High-Performance Automated ML Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/jubito-27/ml-automator/pulls)

**Automac** ek end-to-end Automated Machine Learning (AutoML) library hai jo data preprocessing se lekar model optimization tak ka saara heavy lifting khud karti hai. Isme advanced techniques jaise **Boruta Feature Selection** aur **Optuna-based Parallel Tuning** integrated hain.

---

## 🚀 Key Modules & Features

### 1. 🛡️ Advanced Feature Engineering
Sirf scaling nahi, balki statistically solid feature selection.
* **Boruta Selection:** Shadow features ke saath compete karke irrelevant noise ko hatana.
* **Smart Handling:** Automatic outlier clipping (IQR) aur multicollinearity removal.
* **Encoding:** High-cardinality data ke liye advanced Target Encoding.

### 2. ⚡ Automated Model Tuning
Parallel execution jo aapke CPU ke har core ka sahi istemaal karti hai.
* **Optuna Integration:** Hyperparameter optimization ka gold standard.
* **Smart Allocation:** Cores ko models ke beech distribute karna taaki Windows/Linux dono par maximum speed mile.
* **Supported Models:** XGBoost, LightGBM, CatBoost, RandomForest, SVM, KNN, etc.

### 3. 📝 Text Preprocessing (NLP)
Raw text data ko cleaning aur normalization ke liye ready karna.
* Stopword removal, regex-based tokenization, aur Porter Stemming.

### 4. 📊 Diagnostics & Visualization
Model ko "Black Box" banne se rokna.
* **Learning Curves:** Training vs Validation lines se Overfitting detect karna.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/jubito-27/ml-automator.git](https://github.com/jubito-27/ml-automator.git)
cd ml-automator

# Install in editable mode
pip install -e .
```

