from flask import Flask, request, jsonify
from flask_cors import CORS

from prediction_functions import predict_location


app = Flask(__name__)

CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Landslide AI API is running",
        "message": "XGBoost prediction server is ready"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None:
        return jsonify({
            "error": "No JSON data received"
        }), 400

    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        return jsonify({
            "error": "Latitude and longitude are required"
        }), 400

    try:

        latitude = float(latitude)
        longitude = float(longitude)

    except (TypeError, ValueError):

        return jsonify({
            "error": "Latitude and longitude must be numbers"
        }), 400

    if latitude < -90 or latitude > 90:

        return jsonify({
            "error": "Invalid latitude"
        }), 400

    if longitude < -180 or longitude > 180:

        return jsonify({
            "error": "Invalid longitude"
        }), 400

    try:

        result = predict_location(
            latitude,
            longitude
        )

        if result is None:

            return jsonify({
                "error": "Environmental data unavailable for this location"
            }), 400

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    print("Starting Landslide AI Server...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )