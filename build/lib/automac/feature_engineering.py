import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import itertools

class FeatureImportance:
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
        '''
        You can get all the features above the threshold value (by default 0)
        :param model: Provide the model Name
        :param test_size: size of testing data
        :param random_state: Provide Random State
        :param value: the threshold value on which you want to the columns
        '''
        try:
            important = self.get_feature_importances(model=model , test_size=test_size , random_state=random_state)
            imp_featues = important[important['Importance'] > value]
            return imp_featues
        except Exception as e:
            return f"Error Occured as {e}"

    def remove_multicollinearity(self, threshold=0.9):
        """
        Drops features that are highly correlated with each other.
        """
        corr_matrix = self.x.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if to_drop:
            print(f"✂️ Dropping highly correlated features: {to_drop}")
            self.x = self.x.drop(columns=to_drop)
        return self.x

    def handle_outliers(self, method='clip'):
        """
        Clips outliers using IQR to prevent model distortion.
        """
        num_cols = self.x.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            q1 = self.x[col].quantile(0.25)
            q3 = self.x[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.x[col] = np.clip(self.x[col], lower, upper)
        print("🛠️ Outliers handled using IQR clipping.")
        return self.x

    def fit_all_at_once(self , drop_corr=True , handle_outliers=True ):
        try:
            if drop_corr:
                self.remove_multicollinearity()

            if handle_outliers:
                self.handle_outliers()
        except Exception as e:
            return f"Erro Occured as {e}"


class FeatureEngineering:

    def __init__(self , x , y,):
        self.x = x
        self.y = y

    def encode_categories(self, column_names=None):
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

    def scale_features(self, column_names=None):
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

    def encode_target(self):
        """
        Encodes target variable y.
        """
        try:
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            return self.y, le
        except Exception as e:
            return f"Target Encoding Error: {str(e)}"

    def fit_all_at_once(self, category_columns=None , scale_columns=None , encode_target = True):
        """
        One-stop solution: Encodes, Scales, and returns Feature Importances.
        """
        try:
            print("--- Starting Automated Pipeline ---")
            if encode_target:
                # 1. Target Encoding (y)
                if self.y.dtype == 'object' or self.y.dtype == 'category':
                    print("Step 1: Encoding Target Variable...")
                    self.encode_target()

            # 2. Categorical Encoding (X)
            if category_columns is None:
                cat_cols = self.x.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols:
                    print(f"Step 2: Encoding Categorical Columns: {cat_cols}")
                    self.encode_categories(cat_cols)
            else:
                print(f"Step 2: Encoding Categorical Columns: {cat_cols}")
                self.encode_categories(column_names=category_columns)


            # 3. Numerical Scaling (X)
            if scale_columns is None:
                num_cols = self.x.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    print(f"Step 3: Scaling Numerical Columns: {num_cols}")
                    self.scale_features(num_cols)
            else:
                print(f"Step 3: Scaling Numerical Columns: {num_cols}")
                self.scale_features(column_names=scale_columns)

            print("--- Pipeline Completed Successfully ---")
            return self.x
        except Exception as e:
            return f"Pipeline Failed: {str(e)}"

class AdvancedFeatureEngineering(FeatureEngineering):
    def create_interactions(self, top_n=5):
        """
        Top N features ke beech algebraic interactions (+, -, *, /) banata hai.
        """
        cols = self.x.columns[:top_n]
        for combo in itertools.combinations(cols, 2):
            self.x[f'{combo[0]}_x_{combo[1]}'] = self.x[combo[0]] * self.x[combo[1]]
            self.x[f'{combo[0]}_div_{combo[1]}'] = self.x[combo[0]] / (self.x[combo[1]] + 1e-6)
        print(f"✨ Created interaction features for top {top_n} columns.")
        return self.x

    def recursive_feature_addition(self, model, n_features=10):
        """
        SFS (Sequential Feature Selection): Ek-ek karke best features 
        chunta hai jo model ki performance badhate hain.
        """
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward', n_jobs=-1)
        sfs.fit(self.x, self.y)
        selected_cols = self.x.columns[sfs.get_support()]
        self.x = self.x[selected_cols]
        print(f"🎯 SFS Selected Top {n_features} Features.")
        return self.x

    def apply_pca_features(self, n_components=3):
        """
        Purane features ko delete nahi karta, balki naye PCA components 
        as a feature add karta hai (Latent patterns capture karne ke liye).
        """
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.x)
        for i in range(n_components):
            self.x[f'PCA_Component_{i+1}'] = components[:, i]
        return self.x

    def apply_boruta_selection(self, max_iter=100, perc=80, verbose=0):
        """
        Boruta Algorithm: Shadow features ke saath compete karke
        best features select karta hai.
        perc: Threshold (jitna kam, utne zyada features select honge)
        """
        print("🛡️ Running Boruta Feature Selection... (Wait, this takes time)")
        # 1. Base model (RandomForest is required for Boruta)
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        # 2. Initialize Boruta
        # n_estimators='auto' automatically decides number of trees
        feat_selector = BorutaPy(
            rf, 
            n_estimators='auto', 
            verbose=verbose, 
            max_iter=max_iter, 
            perc=perc, 
            random_state=self.seed
        )

        # 3. Fit Boruta (Numpy array required)
        feat_selector.fit(self.x.values, self.y)

        # 4. Results check
        # confirmed: Jo features definitely important hain
        # tentative: Jo borderline hain
        confirmed_cols = self.x.columns[feat_selector.support_].tolist()
        tentative_cols = self.x.columns[feat_selector.support_weak_].tolist()

        print(f"✅ Boruta Confirmed: {len(confirmed_cols)} features")
        print(f"⚠️ Boruta Tentative: {len(tentative_cols)} features")

        # Update x to only confirmed features
        self.x = self.x[confirmed_cols]
        return self.x, confirmed_cols

    def create_polynomial_features(self, degree=2):
        """
        Non-linear relationships capture karne ke liye 
        Polynomial features (x², xy, etc.) banata hai.
        """
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        # Sirf numeric columns par apply karein
        num_cols = self.x.select_dtypes(include=[np.number]).columns
        poly_data = poly.fit_transform(self.x[num_cols])
        poly_cols = poly.get_feature_names_out(num_cols)
        poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=self.x.index)
        # Original df ke saath merge karein
        self.x = pd.concat([self.x.drop(columns=num_cols), poly_df], axis=1)
        print(f"🧬 Created {len(poly_cols)} Polynomial & Interaction features.")
        return self.x


