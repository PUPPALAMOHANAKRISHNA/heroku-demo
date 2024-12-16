import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# Step 1: Load the dataset efficiently
df = pd.read_csv("train.csv", low_memory=False)

# Step 2: Handle missing values
imputer = SimpleImputer(strategy="mean")
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Convert numerical columns to float32 to save memory
df[numerical_cols] = df[numerical_cols].astype("float32")

# Step 3: Encode categorical columns
df = pd.get_dummies(df, columns=["Gender", "Customer Type", "Type of Travel", "Class"], drop_first=True)
df["satisfaction"] = df["satisfaction"].map({"neutral or dissatisfied": 0, "satisfied": 1})

# Step 4: Define features (X) and target (y)
X = df.drop(["id", "satisfaction"], axis=1)
y = df["satisfaction"]

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest model with optimization
model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Save the model using joblib with compression
joblib.dump(model, "satisfaction_model.pkl", compress=3)
print("Compressed model saved as 'satisfaction_model.pkl'")
