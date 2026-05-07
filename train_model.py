import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('house.csv')

# Handle missing values
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

# Encoding categorical data
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# Split data into X and y
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and encoder
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model and LabelEncoder saved successfully!")
