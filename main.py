"""
Credit Card Dataset Regression Comparison 

This script performs:
1. Data preprocessing and scaling
2. Dimensionality reduction (PCA)
3. Model training using Decision Tree, Random Forest, and SVR
4. Hyperparameter tuning using GridSearchCV
5. Model evaluation using MAE, MSE, and R² metrics
6. Visualization (Actual vs Predicted scatter plots)
"""

# ---------------------- Importing Libraries ----------------------
import pandas as pd
import joblib   
from matplotlib import pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------- Data Loading ----------------------
data = pd.read_csv('creditcard.csv')
print(data.head(10))  # Display first 10 rows


# ---------------------- Feature Selection ----------------------
# Dropping less important columns (V8–V28)
df = data.drop([
    'V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V21', 'V20',
    'V19', 'V18', 'V17', 'V16', 'V15', 'V14', 'V13', 'V12', 'V11',
    'V10', 'V9', 'V8'
], axis=1)

# Drop rows with NaN values in 'Amount'
df.dropna(subset=['Amount'], inplace=True)
print(df.head(10))

 
# ---------------------- Data Preprocessing ----------------------
# Normalize features to range [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
print(scaled_data)


# Separate features and target variable
X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']]
y = df['Amount']


# ---------------------- Dimensionality Reduction ----------------------
# Apply PCA to reduce features to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("PCA shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Visualize PCA projection
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow', edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection of Credit Card Dataset')
plt.show()


# ---------------------- Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# =====================================================================
#  RANDOM FOREST REGRESSOR
# =====================================================================
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred_2 = model.predict(X_test)

# Hyperparameter tuning for Random Forest
param_grid_2 = {
    'n_estimators': [50],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
} 
grid_search_2 = GridSearchCV(
    estimator=model,
    param_grid=param_grid_2,
    cv=2,
    scoring='r2',
    verbose=1
) 
grid_search_2.fit(X_train, y_train) 

print("Best Params (RandomForest):", grid_search_2.best_params_)
print("Best Score (RandomForest):", grid_search_2.best_score_)

# Model evaluation
print("MSE (RandomForest):", mean_squared_error(y_test, y_pred_2))
print("MAE (RandomForest):", mean_absolute_error(y_test, y_pred_2))
print("R² (RandomForest):", r2_score(y_test, y_pred_2))

 

# ---------------------- Visualization ----------------------
# Compare Actual vs Predicted values for all models
models = {
    'Random Forest': y_pred_2,
}

for name, preds in models.items():
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, color='skyblue', edgecolor='k', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2)
    plt.title(f'{name} (Actual vs Predicted)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show() 



filename = 'savemodel.joblib'
joblib.dump(model, 'savemodel.joblib' , compress = 3) 








