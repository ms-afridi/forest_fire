import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application
# Load the Ridge regression model and standard scaler
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('model/scaler.pkl', 'rb'))

# Set a threshold for fire occurrence
FIRE_THRESHOLD = 0.5  # Adjust this threshold as needed

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extract input data from the form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Scale the input data
        new_data_scaled = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Predict the fire weather index
        result = ridge_model.predict(new_data_scaled)

        # Determine if fire will occur based on the threshold
        if result[0] > FIRE_THRESHOLD:
            prediction = "Fire"
        else:
            prediction = "No Fire"

        return render_template('result.html', prediction=prediction)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run("0.0.0.0")
