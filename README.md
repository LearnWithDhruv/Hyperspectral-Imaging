## Project Overview
This project implements a machine learning pipeline to preprocess data, reduce dimensionality using PCA, train predictive models using Random Forest and XGBoost, and evaluate their performance using standard metrics.

## Dataset Information
The dataset used in this project is loaded using Pandas and processed to remove missing values, scale features, and prepare it for machine learning algorithms.

## Data Preprocessing
- **Standardization:** Features are scaled using `StandardScaler` to ensure equal importance.
- **Dimensionality Reduction:** PCA is applied to reduce feature dimensionality while retaining key information.
- **Train-Test Split:** The dataset is split into training and testing subsets for proper model evaluation.

## Model Training
Two machine learning models are trained and compared:
1. **Random Forest Regressor**: A tree-based ensemble method that improves predictive accuracy.
2. **XGBoost Regressor**: A gradient boosting method optimized for performance.

## Model Evaluation
The models are evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures average prediction error.
- **Mean Squared Error (MSE)**: Measures squared prediction error, penalizing large errors more heavily.
- **R-Squared Score (R²)**: Represents how well the model explains variance in the target variable.

## Results & Conclusions
- PCA helps reduce computational cost while maintaining predictive accuracy.
- XGBoost generally outperforms Random Forest in terms of error metrics.
- Proper data preprocessing significantly impacts model performance.

## How to Run the Notebook
1. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
2. Open and run `Task_AI_ML.ipynb` in Jupyter Notebook or Google Colab.
3. Analyze the results and fine-tune models as needed.

## Future Improvements
- Experiment with additional models like Neural Networks.
- Optimize hyperparameters for better accuracy.
- Try feature selection techniques to improve efficiency.

---
This README provides a comprehensive guide to the machine learning workflow followed in the notebook.
