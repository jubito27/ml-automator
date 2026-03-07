import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import (
    train_test_split, cross_validate, learning_curve, validation_curve
)
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    classification_report, log_loss, mean_squared_error, r2_score , mean_absolute_error
)

from sklearn.calibration import calibration_curve

class BaseAnalyser:
    """Base class containing shared utilities for analysis."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

class ClassificationAnalyser(BaseAnalyser):
    """
    A toolset for evaluating and comparing classification models.
    This class provides methods to calculate cross-validation scores, 
    generate ROC curves, learning curves, and calibration plots using Plotly.
    """

    def cross_validator(self, model, cv=10):
        """
        Perform k-fold cross-validation with multiple classification metrics.

        Args:
            model: The scikit-learn compatible classifier instance.
            cv (int): Number of cross-validation folds. Defaults to 10.

        Returns:
            pd.DataFrame: Mean scores for accuracy, precision, recall, F1, and ROC-AUC.
        """
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
            model, self.x, self.y, cv=cv,
            scoring=scoring_metrics, n_jobs=-1, error_score='raise'
        )
        return pd.DataFrame(pd.DataFrame(scores).mean(), columns=['Mean Score'])

    def score_comparison(self, models, test_size=0.3, random_state=42, filename="score_data.csv", save_file=False):
        """
        Compare multiple classifiers on various metrics and save to CSV.

        Args:
            models (list): List of classifier instances.
            test_size (float): Proportion of test data.
            random_state (int): Seed for reproducibility.
            filename (str): Path to save the CSV.
            save_file (bool): Whether to save the result.

        Returns:
            pd.DataFrame: Comparison table of all models.
        """
        rows = []
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        for model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            cv_results = self.cross_validator(model)
            row = {
                'Model Names': type(model).__name__,
                'Accuracy': cv_results.loc['test_acc'][0],
                'F1_Macro': cv_results.loc['test_f1_macro'][0],
                'ROC_AUC': cv_results.loc['test_roc_auc'][0],
                'Log_Loss': log_loss(y_test, model.predict_proba(x_test))
            }
            rows.append(row)
        df = pd.DataFrame(rows).set_index('Model Names')
        if save_file:
            self._ensure_dir('reports')
            try:
                file_path = f"reports/{filename}"
                df.to_csv(file_path)
                print(f"✅ Success: Report saved at {file_path}")
            except PermissionError:
                print(f"❌ Error: Please close '{filename}' if it is open in Excel and try again.")
        return df


    def classification_report_comparison(self, models, filename="classification_report.csv" , save_file=False):
        """
        Analyse and save Classification Report comparison.

        Args:
            models (list): List of fitted or unfitted classifiers.
            save_file (bool): Save the plot as an image.
            filename (str): Path for the image file.

        Returns:
            A .csv file having a comparison of classification report of different models
        """
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
                'Model Names': model_name,
                'Accuracy': report['accuracy'],
                'Precision (Macro)': report['macro avg']['precision'],
                'Recall (Macro)': report['macro avg']['recall'],
                'F1-Score (Macro)': report['macro avg']['f1-score'],
                'F1-Score (Weighted)': report['weighted avg']['f1-score']
            }

            all_model_data.append(model_row)
            print(f"✅ Evaluated: {model_name}")
        comparison_df = pd.DataFrame(all_model_data).set_index('Model Names')
        if save_file:
            self._ensure_dir('reports')
            try:
                file_path = f"reports/{filename}"
                comparison_df.to_csv(file_path)
                print(f"✅ Success: Report saved at {file_path}")
            except PermissionError:
                print(f"❌ Error: Please close '{filename}' if it is open in Excel and try again.")
            except Exception as e:
                print(f"{e}")
        return comparison_df

    def auc_curve_saver(self, models, test_size = 0.3 , random_state=42 , save_file = False , filename="roc_auc_model_comparisons.png"):
        """
        Plot and optionally save ROC-AUC curve comparison.

        Args:
            models (list): List of fitted or unfitted classifiers.
            save_file (bool): Save the plot as an image.
            filename (str): Path for the image file.
        """
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
        if save_file:
            self._ensure_dir('reports')
            try:
                file_path = f"reports/{filename}"
                fig.write_image(file_path)
                print(f"✅ Success: Graph saved at {file_path}")
            except PermissionError:
                print(f"❌ Error: Please close '{filename}' if it is open in Excel and try again.")
        fig.show()

    def caliberation_curve(self, models, test_size = 0.3 , random_state=42 , save_file = False, filename="caliberation_model_comparisons.png"):
        """
        Plot and optionally save Caliberation curve comparison.

        Args:
            models (list): List of fitted or unfitted classifiers.
            save_file (bool): Save the plot as an image.
            filename (str): Path for the image file.
        """
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
            self._ensure_dir('reports')
            try:
                file_path = f"reports/{filename}"
                fig.write_image(file_path)
                print(f"✅ Success: Graph saved at {file_path}")
            except PermissionError:
                print(f"❌ Error: Please close '{filename}' if it is open in Excel and try again.")
            except Exception as e:
                print(f"{e}")
        fig.show()


    def learning_curve_saver(self, models, test_size=0.3, random_state=42,cv = 5 , save_file=False, filename="learning_curve.png", train_set=True):
        """
        Plot and optionally save Learning curve comparison.
        """
        fig = go.Figure()
        train_sizes = np.linspace(0.1, 1.0, 5)

        # FIX: Correct sequence is X_train, X_test, y_train, y_test
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        for model in models:
            model_name = type(model).__name__
            # Choose which data to use for learning curve cross-validation
            data_x, data_y = (x_train, y_train) if train_set else (x_test, y_test)
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, data_x, data_y,
                train_sizes=train_sizes,
                cv=cv,
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
            title=f"Learning Curves Comparison ({'Training' if train_set else 'Test'} Set)",
            xaxis_title='Number of Samples',
            yaxis_title='Accuracy Score',
            template='plotly_white',
            width=900,
            height=600
        )

        if save_file:
            self._ensure_dir('reports')
            try:
                file_path = f"reports/{filename}"
                fig.write_image(file_path)
                print(f"✅ Success: Learning Curve saved at reports/{file_path}")
            except PermissionError:
                print(f"❌ Error: Please close '{filename}' if it is open in Excel and try again.")
            except Exception as e:
                print(f"{e}")
        fig.show()

    def validation_curve_plotter(self, model, param_name, param_range, cv=5, save_file=False, filename="val_curve.png"):
        """
        Plots the validation curve for a specific hyperparameter to analyze Bias-Variance tradeoff.

        Args:
            model: The classifier instance (e.g., RandomForestClassifier()).
            param_name (str): Name of the hyperparameter to vary (e.g., 'max_depth').
            param_range (list or np.array): The values of the parameter to test.
            cv (int): Number of cross-validation folds. Defaults to 5.
            save_file (bool): Whether to save the plot as an image.
            filename (str): Name of the file to save.
        """
        print(f"📊 Calculating Validation Curve for {param_name}...")
        # Validation curve calculate karna
        train_scores, test_scores = validation_curve(
            model, self.x, self.y, 
            param_name=param_name, 
            param_range=param_range,
            cv=cv, 
            scoring="accuracy", 
            n_jobs=-1
        )

        # Mean aur Standard Deviation nikalna
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        fig = go.Figure()

        # Training Score Trace
        fig.add_trace(go.Scatter(
            x=param_range, y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue')
        ))

        # Cross-Validation Score Trace
        fig.add_trace(go.Scatter(
            x=param_range, y=test_mean,
            mode='lines+markers',
            name='Cross-Validation Score',
            line=dict(color='green')
        ))

        fig.update_layout(
            title=f'Validation Curve for {type(model).__name__} ({param_name})',
            xaxis_title=f'Parameter: {param_name}',
            yaxis_title='Accuracy Score',
            template='plotly_white',
            hovermode='x unified'
        )

        if save_file:
            self._ensure_dir('reports')
            try:
                file_path = f"reports/{filename}"
                fig.write_image(file_path)
                print(f"✅ Validation Curve saved at: results/{filename}")
            except PermissionError:
                print(f"❌ Error: Please close '{filename}' if it is open in Excel and try again.")
            except Exception as e:
                print(f"{e}")


        fig.show()

class RegressionAnalyser(BaseAnalyser):
    """
    Dedicated toolset for Regression tasks.
    Includes Error analysis, Residual plots, and Prediction vs Actual comparisons.
    """

    def regression_metrics(self, model, test_size=0.3, random_state=42):
        """
        Calculate standard regression metrics (MSE, RMSE, MAE, R2).

        Args:
            model: Scikit-learn regressor.
            test_size (float): Test split ratio.
            random_state (int): Seed value.

        Returns:
            dict: Dictionary of calculated metrics.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

    def plot_regression_results(self, model, save_file=False, filename="results/regression_analysis.png"):
        """
        Generate two key regression plots: Prediction vs Actual and Residual Plot.

        Args:
            model: Regressor instance.
            save_file (bool): Save image if True.
            filename (str): Output path.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        residuals = y_test - y_pred

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Actual vs Predicted", "Residual Plot"))

        # Plot 1: Actual vs Predicted
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], name='Perfect Fit', line=dict(color='red')), row=1, col=1)

        # Plot 2: Residuals
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

        fig.update_layout(height=500, title_text=f"Analysis: {type(model).__name__}", template="plotly_white")
        if save_file:
            self._ensure_dir(filename)
            fig.write_image(filename)
        fig.show()



    def validation_curve_plotter(self, model, param_name, param_range, cv=5, scoring="r2", save_file=False, filename="results/val_curve.png"):
        """
        Plot Validation Curve to analyze bias-variance tradeoff for a parameter.

        Args:
            model: Regressor or Classifier.
            param_name (str): Parameter to tune.
            param_range (list): Values to test.
            scoring (str): Metric to use (e.g., 'r2' or 'neg_mean_squared_error').
        """
        train_scores, test_scores = validation_curve(
            model, self.x, self.y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=param_range, y=np.mean(train_scores, axis=1), name="Train Score"))
        fig.add_trace(go.Scatter(x=param_range, y=np.mean(test_scores, axis=1), name="Cross-Val Score"))

        fig.update_layout(title=f"Validation Curve ({param_name})", xaxis_title=param_name, yaxis_title=scoring)
        if save_file:
            self._ensure_dir(filename)
            fig.write_image(filename)
        fig.show()