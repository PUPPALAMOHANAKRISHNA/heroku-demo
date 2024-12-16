from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the compressed model and column names
model = joblib.load("satisfaction_model.pkl")

# Load a sample of the training data to extract column names
df = pd.read_csv("train.csv", low_memory=False)

# Process categorical columns to ensure the same features
categorical_columns = ["Gender", "Customer Type", "Type of Travel", "Class"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
model_columns = df.drop(["id", "satisfaction"], axis=1).columns

# Route for the home page
@app.route('/')
def index():
    return render_template("index.html")

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [
            float(request.form.get("Age")),
            float(request.form.get("FlightDistance")),
            float(request.form.get("InflightWifi")),
            float(request.form.get("EaseOfOnlineBooking")),
            float(request.form.get("FoodAndDrink")),
            float(request.form.get("SeatComfort")),
            float(request.form.get("Cleanliness")),
            float(request.form.get("DepartureDelay")),
            float(request.form.get("ArrivalDelay"))
        ]
        
        # Extract categorical features from the form
        categorical_features = [
            request.form.get("Gender"),
            request.form.get("CustomerType"),
            request.form.get("TypeOfTravel"),
            request.form.get("Class")
        ]
        
        # Convert features to DataFrame
        columns = ["Age", "FlightDistance", "InflightWifi", "EaseOfOnlineBooking", 
                   "FoodAndDrink", "SeatComfort", "Cleanliness", "DepartureDelay", "ArrivalDelay"]
        
        input_data = pd.DataFrame([features], columns=columns)
        
        # Add categorical features to DataFrame
        input_data["Gender"] = categorical_features[0]
        input_data["CustomerType"] = categorical_features[1]
        input_data["TypeOfTravel"] = categorical_features[2]
        input_data["Class"] = categorical_features[3]
        
        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, columns=["Gender", "CustomerType", "TypeOfTravel", "Class"], drop_first=True)
        
        # Ensure that the input data has the same columns as the model was trained on
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        
        # Predict satisfaction
        prediction = model.predict(input_data)[0]
        result = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"
        
        return render_template("index.html", prediction=result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
