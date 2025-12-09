import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_country='Saudi Arabia', test_size=0.2):
    
    similar_countries = ['Saudi Arabia', 'Egypt', 'United Arab Emirates', 'Qatar', 'Oman', 'Kuwait', 'Bahrain']
    df_filtered = df[df['Area'].isin(similar_countries)].copy()
    
    if len(df_filtered) < 50:
        print("Warning: Limited data for target countries. Using all data.")
        df_filtered = df.copy()
    
    features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    target = 'hg/ha_yield'
    
    X = pd.get_dummies(df_filtered[['Item'] + features], columns=['Item'], drop_first=True)
    y = df_filtered[target]
    
    try:
        sorted_years = sorted(df_filtered['Year'].unique())
        split_idx = int(len(sorted_years) * 0.8)
        train_years = sorted_years[:split_idx]
        test_years = sorted_years[split_idx:]
        
        X_train = X[X['Year'].isin(train_years)]
        X_test = X[X['Year'].isin(test_years)]
        y_train = y[X['Year'].isin(train_years)]
        y_test = y[X['Year'].isin(test_years)]
        
        if len(X_test) < 5:
            raise ValueError("Not enough test samples")
            
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns