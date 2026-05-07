from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Initialize model and label encoder
model = None
le = None

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
except FileNotFoundError:
    print("Model or LabelEncoder not found. Please run train_model.py first.")

@app.route('/')
def home():
    # Define categorical options for the dropdown
    ocean_options = le.classes_.tolist() if le is not None else []
    return render_template('index.html', ocean_options=ocean_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [
            float(request.form['longitude']),
            float(request.form['latitude']),
            float(request.form['housing_median_age']),
            float(request.form['total_rooms']),
            float(request.form['total_bedrooms']),
            float(request.form['population']),
            float(request.form['households']),
            float(request.form['median_income']),
            request.form['ocean_proximity']
        ]
        
        # Preprocess features
        # Encode ocean_proximity
        features[-1] = le.transform([features[-1]])[0]
        
        # Convert to numpy array for prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)
        
        return render_template('index.html', 
                               prediction_text=f'Estimated House Value: ${output:,}',
                               ocean_options=le.classes_.tolist() if le is not None else [])
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error: {str(e)}',
                               ocean_options=le.classes_.tolist() if le is not None else [])

if __name__ == "__main__":
    app.run(debug=True)
