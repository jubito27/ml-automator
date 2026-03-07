import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score , log_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from concurrent.futures import ThreadPoolExecutor
import warnings

# Clean terminal output
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings('ignore')


class ClassificationTuner:
    """
    Hyperparameter Tuning Engine for Classification tasks using Optuna.
    """

    def __init__(self,x , y ,  n_trials=50,  x_test = 0.2 ,  seed=42 ):
        self.n_trials = n_trials
        self.x = x
        self.y = y
        self.seed = seed
        self.x_test = x_test
        self.best_configs = {} # Final dictionary for all results

    # --- Objective Functions for Each Model ---

    def __splitter(self):
        x_train , x_test , y_train , y_test = train_test_split(self.x , self.y , test_size=self.x_test , random_state=self.seed , stratify=self.y)
        return x_train , x_test , y_train , y_test

    def __dt_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': self.seed
        }
        return self.__evaluate(DecisionTreeClassifier(**params), x_t, y_t, x_v, y_v)

    def __svm_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'probability': True, # Log-loss ke liye probability zaroori hai
            'random_state': self.seed
        }
        return self.__evaluate(SVC(**params), x_t, y_t, x_v, y_v)

    def __knn_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
            'n_jobs': -1
        }
        return self.__evaluate(KNeighborsClassifier(**params), x_t, y_t, x_v, y_v)

    def __ada_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 2.0, log=True),
            'random_state': self.seed
        }
        # AdaBoost boosting model hai par ye eval_set support nahi karta 
        # isliye is_boost=False rahega
        return self.__evaluate(AdaBoostClassifier(**params), x_t, y_t, x_v, y_v, is_boost=False)


    def __rf_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 64),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'n_jobs': -1, 'random_state': self.seed
        }
        return self.__evaluate(RandomForestClassifier(**params), x_t, y_t, x_v, y_v)

    def __xgb_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'random_state': self.seed, 'verbosity': 0
        }
        return self.__evaluate(xgb.XGBClassifier(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def __lgb_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.seed
        }
        return self.__evaluate(lgb.LGBMClassifier(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def __cat_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'verbose': 0, 'random_state': self.seed
        }
        return self.__evaluate(CatBoostClassifier(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def __evaluate(self, model, x_t, y_t, x_v, y_v, is_boost=False, metric='f1'):
        """
        Model ko train aur evaluate karne ka central point.
        Handled: LGBM verbose error and Multi-metric support.
        """
        import lightgbm as lgb

        if is_boost:
            # Check if model is LightGBM to use callbacks instead of verbose
            if isinstance(model, (lgb.LGBMClassifier)):
                model.fit(
                    x_t, y_t,
                    eval_set=[(x_v, y_v)],
                    callbacks=[lgb.log_evaluation(period=0)] # Modern way to silence logs
                )
            else:
                # XGBoost and CatBoost still support verbose=False in many versions
                model.fit(x_t, y_t, eval_set=[(x_v, y_v)], verbose=False)
        else:
            model.fit(x_t, y_t)

        # Metric Logic
        if metric == 'f1':
            preds = model.predict(x_v)
            # 'macro' use kar rahe hain taaki saari classes ko barabar weight mile
            return f1_score(y_v, preds, average='macro')
        elif metric == 'logloss':
            try:
                probs = model.predict_proba(x_v)
                return -log_loss(y_v, probs)
            except AttributeError:
                return -1.0
        else: # accuracy
            preds = model.predict(x_v)
            return accuracy_score(y_v, preds)

    # --- Core Tuning Logic ---
    def __tune_single_model(self, model_key , x_train , y_train , x_val , y_val):
        """
        Kisi ek model ko tune karne ke liye.
        model_key: 'rf', 'xgb', 'lgb', 'cat', 'ada', 'dt', 'svm', 'knn'
        """

        # 1. Available models ki dictionary
        objectives = {
            'rf': ('RandomForest', self.__rf_obj),
            'xgb': ('XGBoost', self.__xgb_obj),
            'lgb': ('LightGBM', self.__lgb_obj),
            'cat': ('CatBoost', self.__cat_obj),
            'ada': ('AdaBoost', self.__ada_obj),
            'dt': ('DecisionTree', self.__dt_obj),
            'svm': ('SVM', self.__svm_obj),
            'knn': ('KNN', self.__knn_obj)
        }

        # 2. Check karein ki user ne sahi key dali hai ya nahi
        if model_key not in objectives:
            available_keys = ", ".join([f"'{k}'" for k in objectives.keys()])
            # VS Code ya terminal mein ye error user ko guide karega
            raise ValueError(
                f"❌ Invalid model_key: '{model_key}'. "
                f"Please choose from the following valid options: {available_keys}"
            )

        # 3. Agar sahi hai toh tuning shuru karein
        name, obj_func = objectives[model_key]
        print(f"\n🚀 Starting Tuning for: {name}")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(lambda t: obj_func(t, x_train, y_train, x_val, y_val), n_trials=self.n_trials)
        print(f"✅ {name} Tuning Complete. Best Accuracy: {study.best_value:.4f}")
        return {name: study.best_params}


    def tune(self, model_keys=['rf', 'xgb', 'lgb', 'cat', 'ada', 'dt', 'svm', 'knn']):
        """
        Runs parallel tuning for classification task

        Returns
            Dictionary of best scores of given models in model_keys
        """
        # 1. Valid keys ki list for suggestion
        all_valid_keys = ['rf', 'xgb', 'lgb', 'cat', 'ada', 'dt', 'svm', 'knn']

        # 2. Validation check
        for key in model_keys:
            if key not in all_valid_keys:
                raise ValueError(
                    f"❌ Invalid model key: '{key}'.\n"
                    f"Available options are: {all_valid_keys}"
                )

        # 3. Data split (aapka internal splitter method)
        x_train, x_val,y_train , y_val = self.__splitter()
        # Purane results clear karein taaki nayi tuning fresh ho
        self.best_configs = {}

        print(f"🚀 Starting parallel tuning for: {model_keys}")

        # 4. Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Note: self._tune_single_model ko call kar rahe hain
            futures = [executor.submit(self.__tune_single_model, key, x_train, y_train, x_val, y_val) for key in model_keys]
            for f in futures:
                result = f.result()
                if result:
                    self.best_configs.update(result)
        print("\n✅ All specified models tuned successfully!")
        return self.best_configs



class RegressionTuner:
    """
    Hyperparameter Tuning Engine for Regression tasks.
    Optimizes for Mean Squared Error (MSE) using Optuna.
    """
    def __init__(self, x, y, n_trials=50, test_size=0.2, seed=42):
        self.x = x
        self.y = y
        self.n_trials = n_trials
        self.test_size = test_size
        self.seed = seed
        self.best_configs = {}

    def __splitter(self):
        # Note: No stratification for regression targets
        return train_test_split(self.x, self.y, test_size=self.test_size, random_state=self.seed)

    def __evaluate(self, model, x_t, y_t, x_v, y_v, is_boost=False):
        """Calculates MSE for Optuna to minimize."""
        if is_boost:
            model.fit(x_t, y_t, eval_set=[(x_v, y_v)], verbose=False)
        else:
            model.fit(x_t, y_t)
        preds = model.predict(x_v)
        return mean_squared_error(y_v, preds)

    # --- Hidden Objective Functions ---

    def __rf_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 800),
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': self.seed
        }
        return self.__evaluate(RandomForestRegressor(**params), x_t, y_t, x_v, y_v)

    def __xgb_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': self.seed, 'verbosity': 0
        }
        return self.__evaluate(xgb.XGBRegressor(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def __lgb_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 256),
            'random_state': self.seed, 'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(x_t, y_t, eval_set=[(x_v, y_v)], callbacks=[lgb.log_evaluation(period=0)])
        return mean_squared_error(y_v, model.predict(x_v))

    def __cat_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'verbose': 0, 'random_seed': self.seed
        }
        return self.__evaluate(CatBoostRegressor(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def __svm_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0), # Regression specific
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        }
        return self.__evaluate(SVR(**params), x_t, y_t, x_v, y_v)

    def __ada_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
            'random_state': self.seed
        }
        return self.__evaluate(AdaBoostRegressor(**params), x_t, y_t, x_v, y_v)

    def __dt_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse'])
        }
        return self.__evaluate(DecisionTreeRegressor(**params), x_t, y_t, x_v, y_v)

    def __knn_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
        }
        return self.__evaluate(KNeighborsRegressor(**params), x_t, y_t, x_v, y_v)

    # --- Core Logic ---

    def __tune_single_model(self, model_key, x_t, y_t, x_v, y_v):
        objectives = {
            'rf': ('RandomForest', self.__rf_obj),
            'xgb': ('XGBoost', self.__xgb_obj),
            'lgb': ('LightGBM', self.__lgb_obj),
            'cat': ('CatBoost', self.__cat_obj),
            'ada': ('AdaBoost', self.__ada_obj),
            'dt': ('DecisionTree', self.__dt_obj),
            'svm': ('SVM', self.__svm_obj),
            'knn': ('KNN', self.__knn_obj)
        }
        name, obj_func = objectives[model_key]
        # Direction 'minimize' for Error
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: obj_func(t, x_t, y_t, x_v, y_v), n_trials=self.n_trials)
        return {name: (study.best_params, study.best_value)}

    def tune(self, model_keys=['rf', 'xgb', 'lgb', 'cat', 'ada', 'dt', 'svm', 'knn']):
        """
        Runs parallel tuning for regression models.

        Returns
            Dictionary of best scores of given models in model_keys
        """
        x_train, x_test, y_train, y_test = self.__splitter()
        self.best_configs = {}

        print(f"🚀 Starting Parallel Regression Tuning for: {model_keys}")

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.__tune_single_model, key, x_train, y_train, x_test, y_test) 
                for key in model_keys
            ]
            for f in futures:
                self.best_configs.update(f.result())

        print(f"✅ Tuning complete!")
        return self.best_configs