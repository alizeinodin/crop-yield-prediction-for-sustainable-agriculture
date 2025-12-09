import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import load_data, preprocess_data
from src.train import train_models
from src.evaluate import evaluate_models, plot_results, compare_with_paper

def main():
    print("="*60)
    print("CROP YIELD PREDICTION PROJECT")
    print("Paper Implementation: Machine Learning for Agriculture")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    print("\n[1] Loading and preprocessing data...")
    df = load_data('data/raw/yield_df.csv')
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    print("\n[2] Training models...")
    models = train_models(X_train, X_test, y_train, y_test)
    
    print("\n[3] Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    results.to_csv('results/metrics.csv', index=False)
    
    print("\n[4] Creating visualizations...")
    plot_results(results)
    
    print("\n[5] Comparing with paper results...")
    comparison = compare_with_paper(results)
    print("\nComparison with Paper:")
    print("-" * 60)
    print(comparison.to_string(index=False))
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    best_model = results.loc[results['R²'].idxmax()]
    worst_model = results.loc[results['R²'].idxmin()]
    
    print(f"Best Model: {best_model['Model']}")
    print(f"  R² Score: {best_model['R²']:.4f}")
    print(f"  RMSE: {best_model['RMSE']:.2f}")
    print(f"  MAE: {best_model['MAE']:.2f}")
    
    print(f"\nWorst Model: {worst_model['Model']}")
    print(f"  R² Score: {worst_model['R²']:.4f}")
    
    print(f"\nAverage R² across all models: {results['R²'].mean():.4f}")
    
    if best_model['Model'] == 'XGBoost':
        print("\n✅ SUCCESS: XGBoost is the best model (matches paper finding)")
    else:
        print(f"\n⚠️ NOTE: {best_model['Model']} performed better than XGBoost")
    
    print(f"\nResults saved to:")
    print(f"  - results/metrics.csv")
    print(f"  - results/plots/comparison.png")
    print(f"  - models/ (trained models)")
    print("="*60)

if __name__ == "__main__":
    main()