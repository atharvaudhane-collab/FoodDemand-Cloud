import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("food_demand_dataset.csv")

# Input features and target
X = df.drop("demand", axis=1)
y = df["demand"]

# Categorical and numerical columns
categorical_features = ["food_item", "day"]
numerical_features = [
    "price",
    "previous_demand",
    "promotion",
    "temperature"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Food Demand Prediction Model Trained Successfully!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Save trained model
joblib.dump(pipeline, "food_demand_model.pkl")

print("Model saved as food_demand_model.pkl")
