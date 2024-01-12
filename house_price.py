# Import necessary libraries
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset
file_path = r'C:\Users\0042H8744\train.csv'
data = pd.read_csv(file_path)

# Drop the target variable from X for imputation
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Create transformers for preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create Ridge regression model with cross-validated GridSearch for optimal alpha (lambda)
ridge = Ridge()
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
ridge_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', ridge_grid)])

# Train Ridge regression model
ridge_model.fit(X, y)

# Print best alpha for Ridge
print("Best alpha for Ridge:", ridge_model.named_steps['regressor'].best_params_['alpha'])

# Create neural network model using MLPRegressor from scikit-learn
nn_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=1)  # Increased max_iter to 500

# Create pipeline for neural network model
nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', nn_model)])

# Train neural network model
nn_pipeline.fit(X, y)

# Evaluate models
ridge_predictions = ridge_model.predict(X)
nn_predictions = nn_pipeline.predict(X)

print("Ridge MSE:", mean_squared_error(y, ridge_predictions))
print("Neural Network MSE:", mean_squared_error(y, nn_predictions))
