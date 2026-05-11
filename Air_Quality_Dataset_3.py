# ============================================================
# ANA500 Micro-Project 3
# Predicting Ozone Levels Using Machine Learning Regression
# Models: Linear Regression and Support Vector Regression
# Dataset: Air Quality Dataset
# ============================================================

# STEP 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# STEP 2: Acquire Data
# ============================================================

# Load dataset from the R datasets repository
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"

df = pd.read_csv(url)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 3: Prepare Data
# ============================================================

# Drop the extra index column if it exists
if "rownames" in df.columns:
    df = df.drop(columns=["rownames"])

# Rename columns for easier use
df = df.rename(columns={
    "Ozone": "ozone",
    "Solar.R": "solar_radiation",
    "Wind": "wind",
    "Temp": "temperature",
    "Month": "month",
    "Day": "day"
})

# Remove rows with missing values
df_clean = df.dropna()

print("\nCleaned Dataset Shape:")
print(df_clean.shape)

print("\nCleaned Dataset Summary:")
print(df_clean.describe())

# Define target variable and predictor variables
X = df_clean[["solar_radiation", "wind", "temperature", "month", "day"]]
y = df_clean["ozone"]

# Split data into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)

# ============================================================
# STEP 4: Explore Data with Visualizations
# ============================================================

# Histogram of ozone levels
plt.figure(figsize=(8, 5))
plt.hist(df_clean["ozone"], bins=20)
plt.title("Distribution of Ozone Levels")
plt.xlabel("Ozone")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("ozone_distribution.png")
plt.show()

# Scatter plot: temperature vs ozone
plt.figure(figsize=(8, 5))
plt.scatter(df_clean["temperature"], df_clean["ozone"])
plt.title("Temperature vs Ozone")
plt.xlabel("Temperature")
plt.ylabel("Ozone")
plt.tight_layout()
plt.savefig("temperature_vs_ozone.png")
plt.show()

# Scatter plot: wind vs ozone
plt.figure(figsize=(8, 5))
plt.scatter(df_clean["wind"], df_clean["ozone"])
plt.title("Wind Speed vs Ozone")
plt.xlabel("Wind Speed")
plt.ylabel("Ozone")
plt.tight_layout()
plt.savefig("wind_vs_ozone.png")
plt.show()

# Correlation matrix
corr = df_clean.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.imshow(corr, interpolation="nearest")
plt.title("Correlation Matrix")
plt.colorbar()

tick_marks = np.arange(len(corr.columns))
plt.xticks(tick_marks, corr.columns, rotation=45)
plt.yticks(tick_marks, corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")

plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# ============================================================
# STEP 5: Analyze Data - Linear Regression Model
# ============================================================

# Create and train linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
linear_predictions = linear_model.predict(X_test)

# Calculate model performance
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

print("\nLinear Regression Results:")
print("MAE:", round(linear_mae, 3))
print("RMSE:", round(linear_rmse, 3))
print("R²:", round(linear_r2, 3))

# Display coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

print("\nLinear Regression Coefficients:")
print(coefficients)

# ============================================================
# STEP 6: Analyze Data - Support Vector Regression Model
# ============================================================

# SVR benefits from scaling, so a pipeline is used
svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])

# Parameter grid for tuning the SVM regression model
param_grid = {
    "svr__kernel": ["rbf", "linear"],
    "svr__C": [1, 10, 100],
    "svr__epsilon": [0.1, 1, 5],
    "svr__gamma": ["scale", "auto"]
}

# Grid search finds the best SVR settings
grid_search = GridSearchCV(
    svr_pipeline,
    param_grid,
    cv=5,
    scoring="r2"
)

grid_search.fit(X_train, y_train)

# Best SVR model
best_svr_model = grid_search.best_estimator_

print("\nBest SVR Parameters:")
print(grid_search.best_params_)

# Make predictions
svr_predictions = best_svr_model.predict(X_test)

# Calculate model performance
svr_mae = mean_absolute_error(y_test, svr_predictions)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_predictions))
svr_r2 = r2_score(y_test, svr_predictions)

print("\nSupport Vector Regression Results:")
print("MAE:", round(svr_mae, 3))
print("RMSE:", round(svr_rmse, 3))
print("R²:", round(svr_r2, 3))

# ============================================================
# STEP 7: Compare Model Performance
# ============================================================

results = pd.DataFrame({
    "Model": ["Linear Regression", "Support Vector Regression"],
    "MAE": [linear_mae, svr_mae],
    "RMSE": [linear_rmse, svr_rmse],
    "R2": [linear_r2, svr_r2]
})

print("\nModel Comparison:")
print(results)

# Save model results to CSV
results.to_csv("model_comparison_results.csv", index=False)

# Bar chart comparing R²
plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["R2"])
plt.title("Model Comparison by R² Score")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.tight_layout()
plt.savefig("model_comparison_r2.png")
plt.show()

# Actual vs predicted for linear regression
plt.figure(figsize=(8, 5))
plt.scatter(y_test, linear_predictions)
plt.title("Linear Regression: Actual vs Predicted Ozone")
plt.xlabel("Actual Ozone")
plt.ylabel("Predicted Ozone")
plt.tight_layout()
plt.savefig("linear_actual_vs_predicted.png")
plt.show()

# Actual vs predicted for SVR
plt.figure(figsize=(8, 5))
plt.scatter(y_test, svr_predictions)
plt.title("SVR: Actual vs Predicted Ozone")
plt.xlabel("Actual Ozone")
plt.ylabel("Predicted Ozone")
plt.tight_layout()
plt.savefig("svr_actual_vs_predicted.png")
plt.show()

# Residual plot for linear regression
linear_residuals = y_test - linear_predictions

plt.figure(figsize=(8, 5))
plt.scatter(linear_predictions, linear_residuals)
plt.axhline(y=0)
plt.title("Linear Regression Residual Plot")
plt.xlabel("Predicted Ozone")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("linear_residual_plot.png")
plt.show()

# ============================================================
# STEP 8: Report and Act
# ============================================================

# Identify best model based on R²
best_model = results.loc[results["R2"].idxmax(), "Model"]

print("\nConclusion:")
print(f"The best performing model was: {best_model}")

print("""
This project used machine learning regression techniques to predict ozone levels
using weather-related variables. The models showed that temperature, wind speed,
and solar radiation can be useful predictors of ozone concentration. These results
could help environmental agencies forecast high-ozone conditions and issue public
health warnings earlier.
""")