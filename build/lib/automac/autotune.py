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
from concurrent.futures import ThreadPoolExecutor # For Parallelism

class AutoTune:
    def __init__(self,x , y ,  n_trials=50,  x_test = 0.2 ,  seed=42):
        self.n_trials = n_trials
        self.x = x
        self.y = y
        self.seed = seed
        self.x_test = x_test
        self.best_configs = {} # Final dictionary for all results

    # --- Objective Functions for Each Model ---

    def _splitter(self):
        x_train , y_train , x_test , y_test = train_test_split(self.x , self.y , self.x_test , random_state=self.seed , stratify=self.y)
        return x_train , y_train , x_test , y_test

    def _dt_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': self.seed
        }
        return self._evaluate(DecisionTreeClassifier(**params), x_t, y_t, x_v, y_v)

    def _svm_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'probability': True, # Log-loss ke liye probability zaroori hai
            'random_state': self.seed
        }
        return self._evaluate(SVC(**params), x_t, y_t, x_v, y_v)

    def _knn_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
            'n_jobs': -1
        }
        return self._evaluate(KNeighborsClassifier(**params), x_t, y_t, x_v, y_v)

    def _ada_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 2.0, log=True),
            'random_state': self.seed
        }
        # AdaBoost boosting model hai par ye eval_set support nahi karta 
        # isliye is_boost=False rahega
        return self._evaluate(AdaBoostClassifier(**params), x_t, y_t, x_v, y_v, is_boost=False)


    def _rf_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 64),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_bool('bootstrap'),
            'n_jobs': -1, 'random_state': self.seed
        }
        return self._evaluate(RandomForestClassifier(**params), x_t, y_t, x_v, y_v)

    def _xgb_obj(self, trial, x_t, y_t, x_v, y_v):
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
        return self._evaluate(xgb.XGBClassifier(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def _lgb_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.seed, 'verbose': -1
        }
        return self._evaluate(lgb.LGBMClassifier(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def _cat_obj(self, trial, x_t, y_t, x_v, y_v):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'verbose': 0, 'random_state': self.seed
        }
        return self._evaluate(CatBoostClassifier(**params), x_t, y_t, x_v, y_v, is_boost=True)

    def _evaluate(self, model, x_t, y_t, x_v, y_v, is_boost=False, metric='accuracy'):
        """
        Model ko train aur evaluate karne ka central point.
        metric: 'accuracy' ya 'logloss'
        """
        if is_boost:
            # Boosting models ke liye eval_set monitor lagana zaroori hai
            model.fit(x_t, y_t, eval_set=[(x_v, y_v)], verbose=False)
        else:
            model.fit(x_t, y_t)
        # Prediction results
        preds = model.predict(x_v)          # Accuracy ke liye labels
        probs = model.predict_proba(x_v)    # Log Loss ke liye probabilities

        if metric == 'logloss':
            # Log Loss ko minimize karna hota hai, par Optuna maximize kar raha hai
            # Isliye hum negative value return karte hain
            return -log_loss(y_v, probs)
        else:
            return accuracy_score(y_v, preds)

    # --- Core Tuning Logic ---
    def _tune_single_model(self, model_key):
        """
        Kisi ek model ko tune karne ke liye.
        model_key: 'rf', 'xgb', 'lgb', 'cat', 'ada', 'dt', 'svm', 'knn'
        """

        x_train , y_train , x_val , y_val = self._splitter()
        # 1. Available models ki dictionary
        objectives = {
            'rf': ('RandomForest', self._rf_obj),
            'xgb': ('XGBoost', self._xgb_obj),
            'lgb': ('LightGBM', self._lgb_obj),
            'cat': ('CatBoost', self._cat_obj),
            'ada': ('AdaBoost', self._ada_obj),
            'dt': ('DecisionTree', self._dt_obj),
            'svm': ('SVM', self._svm_obj),
            'knn': ('KNN', self._knn_obj)
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
        Saare models ko ek saath multithreading se tune karega ya user ke choice ke models.
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
        x_train, y_train, x_val, y_val = self._splitter()
        # Purane results clear karein taaki nayi tuning fresh ho
        self.best_configs = {}

        print(f"🚀 Starting parallel tuning for: {model_keys}")

        # 4. Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Note: self._tune_single_model ko call kar rahe hain
            futures = [executor.submit(self._tune_single_model, key, x_train, y_train, x_val, y_val) for key in model_keys]
            for f in futures:
                result = f.result()
                if result:
                    self.best_configs.update(result)
        print("\n✅ All specified models tuned successfully!")
        return self.best_configs