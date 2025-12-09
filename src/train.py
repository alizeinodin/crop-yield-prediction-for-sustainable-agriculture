import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import joblib

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=140, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=320, random_state=42, n_jobs=-1),
        'DecisionTree': DecisionTreeRegressor(max_depth=20, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=8, n_jobs=-1),
        'Bagging': BaggingRegressor(n_estimators=40, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        trained_models[name] = model
        
        joblib.dump(model, f'models/{name.lower()}_model.pkl')
        print(f"  Saved {name} model")
    
    return trained_models