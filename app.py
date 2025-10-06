from flask import Flask, render_template, request, jsonify
import requests
import joblib
import datetime

app = Flask(__name__)

# Load a pretrained model (create a dummy one if not available)
# You can create a basic model with joblib.dump in another script if needed
try:
    model = joblib.load("model.pkl")
except:
    from sklearn.linear_model import LinearRegression
    import numpy as np
    model = LinearRegression()
    X = np.array([[20, 50, 1000, 3], [25, 60, 1005, 2]])  # temp, humidity, pressure, wind
    y = [0.3, 0.4]  # dummy predictions
    model.fit(X, y)
    joblib.dump(model, "model.pkl")

# Your OpenWeather API key (get from https://openweathermap.org/api)
API_KEY = "d009a31268be23ba67e21fe1d3c6d418"

def get_weather(city):
    """Fetch live weather data using OpenWeather API"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        raise Exception(data.get("message", "Weather data fetch failed."))

    main = data["main"]
    wind = data.get("wind", {})
    features = {
        "temp": main["temp"],
        "humidity": main["humidity"],
        "pressure": main["pressure"],
        "wind_speed": wind.get("speed", 0)
    }
    return data, features

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.json.get("city")
        if not city:
            return jsonify({"error": "City name required"}), 400

        raw, features = get_weather(city)
        X = [[
            features["temp"],
            features["humidity"],
            features["pressure"],
            features["wind_speed"]
        ]]
        prediction = model.predict(X)[0]

        return jsonify({
            "city": city,
            "prediction": float(prediction),
            "units": "CO₂ (predicted)",
            "timestamp": datetime.datetime.now().isoformat(),
            "fetched": features
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
