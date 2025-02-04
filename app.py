from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the dataset
try:
    df = pd.read_csv("Heart Diesase Prediciton\heart.csv")
    X = df[['age', 'cp', 'thalach']]
    Y = df['target']
except FileNotFoundError:
    print("Error: Dataset not found. Please ensure 'heart - heart.csv' is in the project directory.")
    exit()

# Train the logistic regression model
try:
    model = LogisticRegression(max_iter=1000)  # Ensure convergence
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    model.fit(X_train, Y_train)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None

    if request.method == "POST":
        try:
            # Get input from the user
            age = int(request.form.get("age", -1))
            cp = int(request.form.get("cp", -1))
            thalach = int(request.form.get("thalach", -1))

            # Validate user input
            if age < 0 or cp < 0 or thalach < 0:
                raise ValueError("Inputs must be positive integers.")

            # Make prediction
            user_data = np.array([[age, cp, thalach]])
            prediction = model.predict(user_data)[0]
            result = "Heart Disease" if prediction == 1 else "No Heart Disease"

        except ValueError as ve:
            error_message = f"Input error: {ve}"
        except Exception as e:
            error_message = f"An error occurred: {e}"

    return render_template("index.html", result=result, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
