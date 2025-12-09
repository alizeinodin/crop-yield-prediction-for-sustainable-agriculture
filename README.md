# 🌱 Crop Yield Prediction for Sustainable Agriculture

## 📋 Project Overview
Implementation of the research paper: **"Crop yield prediction through machine learning: A path towards sustainable agriculture and climate resilience in Saudi Arabia"** (AIMS Agriculture and Food, 2024).

This project develops machine learning models to predict crop yields based on climatic and agricultural factors, supporting sustainable farming practices and climate resilience in arid regions.

## 🎯 Project Objectives
- Implement ML models for crop yield prediction
- Compare model performance with original paper
- Provide actionable insights for crop rotation strategies
- Support climate-resilient agriculture in Saudi Arabia


## Implementation Details
### Data Source
- **Crop Yield Prediction Dataset** from Kaggle
- Contains data for 101 countries (1990-2013)
- 10 major crops: maize, potatoes, rice, sorghum, soybeans, wheat, cassava, sweet potatoes, plantains, yams
- Features: Area, Item, Year, Average Temperature, Pesticides (tons), Average Rainfall

### Preprocessing
1. Filtered data for Saudi Arabia and similar MENA region countries
2. One-Hot Encoding for categorical variables (Item)
3. Min-Max scaling for numerical features
4. Temporal train-test split (1990-2008 for training, 2009-2013 for testing)

### Machine Learning Models
Implemented five models with hyperparameters from the paper:
1. **Random Forest Regressor** (n_estimators=140)
2. **XGBoost Regressor** (n_estimators=320)
3. **Decision Tree Regressor** (max_depth=20)
4. **K-Nearest Neighbors** (n_neighbors=8)
5. **Bagging Regressor** (n_estimators=40)

### Evaluation Metrics
- R² Score (Coefficient of Determination)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## Results
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Random Forest | 0.8781 | 22279.31 | 12350.53 |
| XGBoost | 0.8806 | 22150.42 | 12289.17 |
| Decision Tree | 0.9456 | 15073.28 | 9876.45 |
| KNN | 0.9032 | 19345.36 | 11023.71 |
| Bagging | 0.8950 | 20012.54 | 11567.82 |

**Best Model:** Decision Tree Regressor (R² = 0.9456)

## Comparison with Original Paper
| Model | Our R² | Paper R² | Difference |
|-------|--------|----------|------------|
| Random Forest | 0.8781 | 0.9740 | -9.8% |
| XGBoost | 0.8806 | 0.9745 | -9.6% |
| Decision Tree | 0.9456 | 0.9628 | -1.8% |
| KNN | 0.9032 | 0.9573 | -5.7% |
| Bagging | 0.8950 | 0.9738 | -8.1% |

## Key Findings
1. The Decision Tree model performed best in our implementation (R² = 0.9456)
2. All models achieved R² > 0.87, demonstrating reasonable predictive capability
3. Our results are lower than the paper's reported values, likely due to:
   - Differences in data preprocessing and feature engineering
   - Limited Saudi Arabia-specific data samples
   - Variations in train-test splitting methodology
4. Contrary to the paper which found XGBoost as the best model, our implementation showed Decision Tree performing best

## Limitations
1. **Data Limitations**: The dataset contains limited samples for Saudi Arabia specifically
2. **Feature Scope**: The implementation uses basic features; the paper may have used additional agricultural parameters
3. **Computational Constraints**: Hyperparameter tuning was limited compared to the original research

## How to Run
1. Install dependencies:
   ```bash
   sudo apt install python3
   ```
   ```bash
   pip install -r requirements.txt
   ```
2. Download dataset from [Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
3. Export it in ``data/raw`` directory of project and ensure ``yield_df.csv`` exists
4. Run project
    ```bash
    python3 ./main.py
    ```