from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Sample training data
data = {
    'study_hours': [2, 4, 6, 8, 10, 1, 3, 7, 5, 9],
    'attendance': [60, 70, 80, 90, 95, 55, 65, 85, 75, 92],
    'previous_score': [50, 60, 70, 80, 90, 45, 55, 75, 65, 88],
    'performance': ['Poor', 'Average', 'Good', 'Good', 'Good', 'Poor', 'Average', 'Good', 'Average', 'Good']
}

df = pd.DataFrame(data)

X = df[['study_hours', 'attendance', 'previous_score']]
y = df['performance']

model = GaussianNB()
model.fit(X, y)
pickle.dump(model, open('model.pkl', 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    features = [float(request.form['study_hours']),
                float(request.form['attendance']),
                float(request.form['previous_score'])]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text=f"Predicted Performance: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
