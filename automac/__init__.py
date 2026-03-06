# Version definition
__version__ = "0.1.1"

# Module se classes aur functions ko expose karna
from .feature_engineering import FeatureImportance , FeatureEngineering
from .autotune import AutoTune
from .analyser import Analyser

# Ye define karta hai ki 'from automater import *' karne par kya kya aayega
__all__ = [
    'FeatureImportance',
    'FeatureEngineering',
    'Analyser',
    'AutoTune'
]