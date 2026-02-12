import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

class FeatureEvaluation:
    def __init__(self, x, y):
        # DataFrame ensure karna
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        self.y = y

    def get_feature_importances(self, model, test_size=0.3, random_state=42):
        """
        Supports both Tree-based (feature_importances_) and Linear models (coef_).
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
            model.fit(X_train, y_train)

            # Check for feature_importances_ (Trees) or coef_ (Linear Models)
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Multiclass ke liye coef_ ka absolute mean lete hain
                imp = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                raise AttributeError(f"Model {type(model).__name__} has no importance attribute.")

            df = pd.DataFrame({
                "Feature": self.x.columns,
                "Importance": imp
            }).sort_values(by="Importance", ascending=False)
            return df
        except Exception as e:
            return f"Error in Feature Importance: {str(e)}"
    def extract_important_feature(self , model, test_size=0.3, random_state=42 , value=0):
        try:
            important = self.feature_importances(model=model , test_size=test_size , random_state=random_state)
            imp_featues = important[important['Importance'] > value]
            return imp_featues
        except Exception as e:
            return f"Error Occured as {e}"

    def auto_encode_categories(self, column_names=None):
        """
        Automatically handles high cardinality using Target Encoding.
        """
        if column_names is None:
            column_names = self.x.select_dtypes(include=['object', 'category']).columns.tolist()
        try:
            encoder = TargetEncoder()
            # Handling Missing Values before encoding
            imputer = SimpleImputer(strategy='most_frequent')
            X_temp = self.x.copy()
            X_temp[column_names] = imputer.fit_transform(X_temp[column_names])
            X_temp[column_names] = encoder.fit_transform(X_temp[column_names], self.y)
            self.x = X_temp
            return self.x, encoder
        except Exception as e:
            return f"Encoding Error: {str(e)}"

    def auto_scale_features(self, column_names=None):
        """
        Scales numerical features. Auto-detects if column_names not provided.
        """
        if column_names is None:
            column_names = self.x.select_dtypes(include=[np.number]).columns.tolist()
        try:
            scaler = StandardScaler()
            imputer = SimpleImputer(strategy='median')
            X_temp = self.x.copy()
            X_temp[column_names] = imputer.fit_transform(X_temp[column_names])
            X_temp[column_names] = scaler.fit_transform(X_temp[column_names])
            self.x = X_temp
            return self.x, scaler
        except Exception as e:
            return f"Scaling Error: {str(e)}"

    def label_encode_target(self):
        """Encodes target variable y."""
        try:
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            return self.y, le
        except Exception as e:
            return f"Target Encoding Error: {str(e)}"

    def fit_all_at_once(self, model, target_col_name=None):
        """
        One-stop solution: Encodes, Scales, and returns Feature Importances.
        """
        try:
            print("--- Starting Automated Pipeline ---")
            # 1. Target Encoding (y)
            if self.y.dtype == 'object' or self.y.dtype == 'category':
                print("Step 1: Encoding Target Variable...")
                self.label_encode_target()

            # 2. Categorical Encoding (X)
            cat_cols = self.x.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                print(f"Step 2: Encoding Categorical Columns: {cat_cols}")
                self.auto_encode_categories(cat_cols)

            # 3. Numerical Scaling (X)
            num_cols = self.x.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                print(f"Step 3: Scaling Numerical Columns: {num_cols}")
                self.auto_scale_features(num_cols)

            # 4. Feature Importance
            print("Step 4: Training Model & Extracting Importances...")
            importance_df = self.get_feature_importances(model)
            print("--- Pipeline Completed Successfully ---")
            return self.x, importance_df
        except Exception as e:
            return f"Pipeline Failed: {str(e)}"


