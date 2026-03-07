# Version definition
__version__ = "0.1.1"

# Module se classes aur functions ko expose karna
from .feature_engineering import FeatureImportance , FeatureEngineering , AdvancedFeatureEngineering
from .autotune import ClassificationTuner , RegressionTuner
from .analyser import ClassificationAnalyser , RegressionAnalyser
from .text import transform

# Ye define karta hai ki 'from automater import *' karne par kya kya aayega
__all__ = [
    'FeatureImportance',
    'FeatureEngineering',
    'AdvancedFeatureEngineering',
    'RegressionAnalyser',
    'ClassificationAnalyser',
    'RegressionTuner',
    'ClassificationTuner',
    'transform'
]