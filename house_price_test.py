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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create Ridge regression model with cross-validated GridSearch for optimal alpha (lambda)
ridge = Ridge()
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
ridge_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', ridge_grid)])

# Train Ridge regression model on the training set
ridge_model.fit(X_train, y_train)

# Evaluate Ridge model on the test set
ridge_predictions = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge MSE on Test Set:", ridge_mse)

# Create neural network model using MLPRegressor from scikit-learn
nn_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=1)  # Increased max_iter to 500

# Create pipeline for neural network model
nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', nn_model)])

# Train neural network model on the training set
nn_pipeline.fit(X_train, y_train)

# Evaluate Neural Network model on the test set
nn_predictions = nn_pipeline.predict(X_test)
nn_mse = mean_squared_error(y_test, nn_predictions)
print("Neural Network MSE on Test Set:", nn_mse)
