import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb


def randomforest(x_train, y_train, x_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"RandomForest - Best Accuracy: {study.best_value:.4f}")
    print(f"RandomForest - Best Parameters: {study.best_params}\n")
    return study.best_params


def adaboost(x_train, y_train, x_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
            'random_state': 42
        }
        model = AdaBoostClassifier(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"AdaBoost - Best Accuracy: {study.best_value:.4f}")
    print(f"AdaBoost - Best Parameters: {study.best_params}\n")
    return study.best_params


def catboost(x_train, y_train, x_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': 0,
            'random_state': 42
        }
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)
        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"CatBoost - Best Accuracy: {study.best_value:.4f}")
    print(f"CatBoost - Best Parameters: {study.best_params}\n")
    return study.best_params


def xgboost(x_train, y_train, x_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"XGBoost - Best Accuracy: {study.best_value:.4f}")
    print(f"XGBoost - Best Parameters: {study.best_params}\n")
    return study.best_params


def lightgbm(x_train, y_train, x_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"LightGBM - Best Accuracy: {study.best_value:.4f}")
    print(f"LightGBM - Best Parameters: {study.best_params}\n")
    return study.best_params


def gradientboosting(x_train, y_train, x_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
        model = GradientBoostingClassifier(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"GradientBoosting - Best Accuracy: {study.best_value:.4f}")
    print(f"GradientBoosting - Best Parameters: {study.best_params}\n")
    return study.best_params


# Usage example
if __name__ == "__main__":
    # Assuming you have x_train, y_train, x_val, y_val already split
    import pandas as pd


    df = pd.read_excel(r"C:\Users\Abhishek sharma\Artificial Intelligence\Datasets\Influenza_surveillance_Data.xlsx")
    from sklearn.model_selection import train_test_split
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    # Optimize all models
    rf_params = randomforest(x_train, y_train, x_val, y_val, n_trials=50)
    ada_params = adaboost(x_train, y_train, x_val, y_val, n_trials=50)
    cat_params = catboost(x_train, y_train, x_val, y_val, n_trials=50)
    xgb_params = xgboost(x_train, y_train, x_val, y_val, n_trials=50)
    lgb_params = lightgbm(x_train, y_train, x_val, y_val, n_trials=50)
    gb_params = gradientboosting(x_train, y_train, x_val, y_val, n_trials=50)
    # Store all best parameters
    all_params = {
        'RandomForest': rf_params,
        'AdaBoost': ada_params,
        'CatBoost': cat_params,
        'XGBoost': xgb_params,
        'LightGBM': lgb_params,
        'GradientBoosting': gb_params
    }
    print("\n=== All Best Parameters ===")
    for model_name, params in all_params.items():
        print(f"{model_name}: {params}")
