# Libraries
import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import pickle

script_dir = Path(__file__).parent
data_file_path = script_dir.parent / 'data' / 'insurance.csv'

df = pd.read_csv(data_file_path)

# defining X,y 
X = df.drop('charges', axis=1)
y = df['charges']

# numberical vs categorical 
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# transforming features 
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# Boosting Model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) # this is a regression model that uses the squared error as the loss function, reg:squarederror is the objective function for regression tasks in XGBoost, and random_state is set for reproducibility

# create pipeline
full_pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('regressor', model)
    ]
)

# Grid Search

# parameters 
param_grid = {
    'regressor__n_estimators': [100, 200], # this is the number of trees that will be built
    'regressor__max_depth': [3, 5, 7], # this is the maximum depth of the trees
    'regressor__learning_rate': [0.01, 0.1, 0.2] # this is the step size shrinkage used in update to prevent overfitting
}

# create grid search

grid_search = GridSearchCV(
    full_pipeline, 
    param_grid,
    cv=5, # number of cross-validation folds, in other words, how many times the model will be trained and tested on different subsets of the data
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, # number of jobs to run in parallel, -1 means using all processors
    verbose=1 
) 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Making predictions

best_model = grid_search.best_estimator_ # best_estimator_ returns the best model found during the grid search
predictions = best_model.predict(X_test) # here we use the best model to make predictions on the test set

rmse = np.sqrt(mean_squared_error(y_test, predictions)) # calculate the mean squared error
r2 = r2_score(y_test, predictions) # calculate the r2 score
print(f"RMSE: {rmse}")
print(f"R2: {r2}")


# Saving the model with pickle
model_save_path = script_dir / 'premium_predictor_model.pkl'

with open(model_save_path, 'wb') as file:
    pickle.dump(best_model, file)

print(f'\n Model saved as {model_save_path}')
