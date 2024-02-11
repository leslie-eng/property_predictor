from flask import Flask, render_template
import pickle
import numpy as np
import requests
import pandas as pd

# response = requests.get(url="")
# response.raise_for_status()

# data = response.json{}
# longitude = int(data[""])
# latitude = int(data[""])

model = pickle.load(open('<file name>', 'rb'))

app = Flask(__name__)

@app.route('', methods=['POST'])
def make_prediction(area, lat, lon, neighborhood):
    data = {
        "surface_covered_in_m2":area,
        "lat":lat,
        "lon":lon,
        "neighborhood":neighborhood,
    }
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df).round(2)[0]
    return render_template('price/price.html')


if __name__ == '__main__':
    app.run(debug=True)