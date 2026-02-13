# Version definition
__version__ = "0.1.0"

# Module se classes aur functions ko expose karna
from .feature_engineering import FeatureEvaluation
from .models_optimizer import (
    randomforest, 
    adaboost, 
    catboost, 
    xgboost, 
    lightgbm, 
    gradientboosting
)
from .trainer import Trainer

# Ye define karta hai ki 'from automater import *' karne par kya kya aayega
__all__ = [
    'FeatureEvaluation',
    'Trainer',
    'randomforest',
    'adaboost',
    'catboost',
    'xgboost',
    'lightgbm',
    'gradientboosting'
]
