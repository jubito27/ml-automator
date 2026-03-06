from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_validate
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import pandas as pd
from sklearn.metrics import classification_report , log_loss
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
import os

class Analyser:
    def __init__(self , x , y):
        self.x = x
        self.y = y

    def cross_validator(self , model, cv=10):
        scoring_metrics = {
            'acc': 'accuracy',
            'bal_acc': 'balanced_accuracy',
            'prec_macro': 'precision_macro',
            'rec_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted',
            'roc_auc': 'roc_auc'
        }
        scores = cross_validate(
            model,
            self.x,
            self.y,
            cv=cv,
            scoring=scoring_metrics,
            n_jobs=1,  # Force single process
            error_score='raise' # Isse pata chalega asli error kya hai
        )
        df_results = pd.DataFrame(scores)
        df = pd.DataFrame(df_results.mean())
        return df

    def model_scores(self , model , test_size = 0.3 , random_state=42 , cv=10):
        x_train , x_text , y_train , y_test = train_test_split(self.x , self.y , test_size=test_size , random_state=random_state , stratify=self.y)
        model.fit(x_train , y_train)
        y_pred = model.predict(x_text)
        #scorings
        train_score = model.score(x_train , y_train)
        text_score = model.score(x_text , y_test)
        log_loss_ = log_loss(y_test , y_pred)
        c_v = self.cross_validator(model , cv=10)
        return c_v , train_score , text_score , log_loss_

    def score_comparison(self , models , test_size = 0.3 , random_state=42 , filename="score_data.csv" , save_file=True):
        rows = []
        for model in models:
            c_v, train_score_, text_score_, log_loss_ = self.model_scores(model, test_size=test_size, random_state=random_state , stratify=self.y)
            row = {
                'Model Name': type(model).__name__,
                'fit_time': c_v.loc['fit_time'][0],
                'train_score': train_score_,
                'test_score': text_score_,
                'accuracy': c_v.loc['test_acc'][0],
                'precision_macro': c_v.loc['test_prec_macro'][0],
                'recall_macro': c_v.loc['test_rec_macro'][0],
                'f1_macro': c_v.loc['test_f1_macro'][0],
                'f1_weighted': c_v.loc['test_f1_weighted'][0],
                'roc_auc': c_v.loc['test_roc_auc'][0], # Fixed
                'log_loss': log_loss_
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if save_file:
            if not os.path.exists('reports'):
                os.makedirs('reports')
            file_path = f"reports/{filename}"
            df.to_csv(file_path)
            print(f"\nModel Comparison score Data saved at: {file_path}")
        return df


    def classification_report_comparison(self, models, filename="classification_report.csv" , save_file=True):
        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.3,
            random_state=42, stratify=self.y
        )
        all_model_data = []

        for model in models:
            model_name = type(model).__name__
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            model_row = {
                'Model Name': model_name,
                'Accuracy': report['accuracy'],
                'Precision (Macro)': report['macro avg']['precision'],
                'Recall (Macro)': report['macro avg']['recall'],
                'F1-Score (Macro)': report['macro avg']['f1-score'],
                'F1-Score (Weighted)': report['weighted avg']['f1-score']
            }

            all_model_data.append(model_row)
            print(f"✅ Evaluated: {model_name}")
        comparison_df = pd.DataFrame(all_model_data)
        if save_file:
            if not os.path.exists('reports'):
                os.makedirs('reports')
            file_path = f"reports/{filename}"
            comparison_df.to_csv(file_path)
            print(f"\nFinal Comparison Table saved at: {file_path}")
        return comparison_df

    def auc_curve_saver(self, models, test_size = 0.3 , random_state=42 , save_file = False , filename="roc_auc_model_comparisons.png"):
        X_train , X_test , y_train , y_test = train_test_split(self.x , self.y , test_size=test_size , random_state=random_state , stratify=self.y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.5)',
            line=dict(dash='dash', color='grey')
        ))
        for model in models:
            model_name = type(model).__name__
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC: {auc_score:.2f})'
            ))

        fig.update_layout(
            title='ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white'
        )
        file_path = f"results/{filename}"
        if save_file:
            if not os.path.exists('results'):
                os.makedirs('results')

            fig.write_image(file_path)
            print(f"✅ Success: Graph saved at {file_path}")
        fig.show()

    def caliberation_curve(self, models, test_size = 0.3 , random_state=42 , save_file = False, filename="caliberation_model_comparisons.png"):
        X_train , X_test , y_train , y_test = train_test_split(self.x , self.y , test_size=test_size , random_state=random_state , stratify=self.y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.5)',
            line=dict(dash='dash', color='grey')
        ))
        for model in models:
            model_name = type(model).__name__
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            prob_true , prob_pred = calibration_curve(y_test , y_proba)
            fig.add_trace(go.Scatter(
                x=prob_pred, y=prob_true,
                mode='lines',
                name=f'{model_name})'
            ))

            fig.update_layout(
            title='Caliberation Curve Comparison',
            xaxis_title='Predicted Probability',
            yaxis_title='True Probability',
            template='plotly_white'
            )
        if save_file:
            file_path = f"results/{filename}"
            if not os.path.exists('results'):
                os.makedirs('results')

                fig.write_image(file_path)
                print(f"Success: Graph saved at {file_path}")
        fig.show()

    def learning_curve(self, models, test_size = 0.3 , random_state=42 , save_file = False , filename="learning_curve.png"):
        fig = go.Figure()
        train_sizes = np.linspace(0.1, 1.0, 5)

        for model in models:
            model_name = type(model).__name__
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, self.x, self.y,
                train_sizes=train_sizes,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            test_scores_mean = np.mean(test_scores, axis=1)
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=test_scores_mean,
                mode='lines+markers',
                name=f'{model_name}'
            ))
        fig.update_layout(
            title='Learning Curves Comparison (Validation Scores)',
            xaxis_title='Number of Training Samples',
            yaxis_title='Accuracy Score',
            template='plotly_white',
            width=900,
            height=600
        )
        if save_file:
            if not os.path.exists('reports'):
                os.makedirs('reports')
            file_path = f"reports/{filename}"
            fig.write_image(file_path)
            print(f"Multi-Model Learning Curve saved at: {file_path}")
        fig.show()



# if __name__ == "__main__":
#     df = pd.read_excel(r"C:\Users\Abhishek sharma\Artificial Intelligence\Datasets\Influenza_surveillance_Data.xlsx")

#     x = df.iloc[:,1:]
#     y = df.iloc[:,0]
#     x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3 , random_state=42 , stratify=y)

#     import models_optimizer as optuna

#     rf = optuna.randomforest(x_train=x_train , y_train=y_train , x_val=x_test , y_val=y_test)

#     ab = optuna.adaboost(x_train=x_train , y_train=y_train , x_val=x_test , y_val=y_test)

#     cb = optuna.catboost(x_train=x_train , y_train=y_train , x_val=x_test , y_val=y_test)

#     xb = optuna.xgboost(x_train=x_train , y_train=y_train , x_val=x_test , y_val=y_test)

#     import lightgbm as lgb
#     _orig_lgbm_fit = lgb.LGBMClassifier.fit
#     def _lgbm_fit_no_verbose(self, X, y, *args, **kwargs):
#         kwargs.pop('verbose', None)
#         return _orig_lgbm_fit(self, X, y, *args, **kwargs)
#     lgb.LGBMClassifier.fit = _lgbm_fit_no_verbose

    # lg = optuna.lightgbm(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)

    # from catboost import CatBoostClassifier
    # import xgboost as xgb
    # import lightgbm as lgb
    # from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    # models = [RandomForestClassifier(**rf) , AdaBoostClassifier(**ab) , CatBoostClassifier(**cb) , xgb.XGBClassifier(**xb) , lgb.LGBMClassifier(**lg)]
    # train = Trainer(x=x , y=y)

    # frame = train.score_comparison(models=models)
    # print(frame)

