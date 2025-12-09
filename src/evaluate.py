import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_models(models, X_test, y_test):
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            'Model': name,
            'R²': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2)
        })
        
        print(f"{name}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    
    return pd.DataFrame(results)

def plot_results(results_df):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(results_df['Model'], results_df['R²'])
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, results_df['R²']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(results_df['Model'], results_df['RMSE'])
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/plots/comparison.png', dpi=300)
    plt.show()
    
    return results_df

def compare_with_paper(our_results):
    paper_results = {
        'RandomForest': {'R²': 0.9740, 'RMSE': 13781.94, 'MAE': 3076.19},
        'XGBoost': {'R²': 0.9745, 'RMSE': 15803.15, 'MAE': 2681.33},
        'DecisionTree': {'R²': 0.9628, 'RMSE': 16697.12, 'MAE': 3751.65},
        'KNN': {'R²': 0.9573, 'RMSE': 17603.76, 'MAE': 4650.21},
        'Bagging': {'R²': 0.9738, 'RMSE': 14010.27, 'MAE': 3294.23}
    }
    
    comparison = []
    
    for idx, row in our_results.iterrows():
        model = row['Model']
        if model in paper_results:
            paper_r2 = paper_results[model]['R²']
            diff = row['R²'] - paper_r2
            diff_percent = (diff / paper_r2) * 100
            
            comparison.append({
                'Model': model,
                'Our_R²': row['R²'],
                'Paper_R²': paper_r2,
                'Difference': f"{diff:+.4f}",
                'Diff_Percent': f"{diff_percent:+.1f}%",
                'Status': 'Better' if diff > 0 else 'Worse'
            })
    
    return pd.DataFrame(comparison)