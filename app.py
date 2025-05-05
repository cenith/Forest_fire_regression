import pickle
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline (which includes scaler + linear regression)
model = pickle.load(open("lr_pipeline.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Extract input values from the form
        features = [
            float(request.form["Temperature"]),
            float(request.form["RH"]),
            float(request.form["Ws"]),
            float(request.form["Rain"]),
            float(request.form["FFMC"]),
            float(request.form["DMC"]),
            float(request.form["ISI"]),
            int(request.form["Classes"]),
            int(request.form["Region"])
        ]

        # Convert to 2D array for prediction
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]

        # Round for neat display
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


