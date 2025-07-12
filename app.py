from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form
    features = [
        float(request.form["f1"]),
        float(request.form["f2"]),
        float(request.form["f3"]),
        float(request.form["f4"]),
        float(request.form["f5"]),
        float(request.form["f6"]),
        float(request.form["f7"]),
        float(request.form["f8"]),
    ]
    
    # Scale and predict
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
    return render_template("result.html", prediction=result)

# API Endpoints
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for diabetes prediction
    Expects JSON data with features: f1, f2, f3, f4, f5, f6, f7, f8
    Returns JSON response with prediction result
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract features from JSON
        required_features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        features = []
        
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
            
            try:
                features.append(float(data[feature]))
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {feature}. Must be a number."}), 400
        
        # Scale and predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        
        return jsonify({
            "prediction": result,
            "prediction_code": int(prediction),
            "confidence": {
                "non_diabetic": float(probability[0]),
                "diabetic": float(probability[1])
            },
            "features": dict(zip(required_features, features))
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/predict", methods=["GET"])
def api_predict_get():
    """
    API endpoint for diabetes prediction via GET request
    Expects query parameters: f1, f2, f3, f4, f5, f6, f7, f8
    Returns JSON response with prediction result
    """
    try:
        # Get query parameters
        required_features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        features = []
        
        for feature in required_features:
            if feature not in request.args:
                return jsonify({"error": f"Missing required parameter: {feature}"}), 400
            
            try:
                features.append(float(request.args[feature]))
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {feature}. Must be a number."}), 400
        
        # Scale and predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        
        return jsonify({
            "prediction": result,
            "prediction_code": int(prediction),
            "confidence": {
                "non_diabetic": float(probability[0]),
                "diabetic": float(probability[1])
            },
            "features": dict(zip(required_features, features))
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/health", methods=["GET"])
def api_health():
    """
    Health check endpoint
    Returns basic API status
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "endpoints": {
            "predict_post": "/api/predict (POST)",
            "predict_get": "/api/predict (GET)",
            "health": "/api/health (GET)"
        }
    })

if __name__ == "__main__":
    app.run(debug=False)
