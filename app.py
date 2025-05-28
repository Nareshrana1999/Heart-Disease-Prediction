import os
import sys
import warnings
import logging
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from data_preprocessing import load_data

# Suppress all warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Load models and scaler
MODELS_DIR = 'models'
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Feature names in the exact order expected by the models
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting electrocardiographic results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment (0-2)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)'
}

# Load models and scaler
models = {}
scaler = None

try:
    # Load scaler
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("Loaded scaler successfully")
    
    # Load models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.pkl')]
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '')
        model_path = os.path.join(MODELS_DIR, model_file)
        models[model_name] = joblib.load(model_path)
        print(f"Loaded model: {model_name}")
        
except Exception as e:
    print(f"Error loading models: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html', 
                         feature_names=FEATURE_NAMES,
                         feature_descriptions=FEATURE_DESCRIPTIONS)

def calculate_risk_factors(features):
    """Calculate risk factors based on input features"""
    risk_factors = []
    risk_score = 0
    
    # Age
    age = features[0]
    if age > 50:
        risk_factors.append(f'Age > 50 ({int(age)} years)')
        risk_score += 0.2
    
    # Sex (male is higher risk)
    if features[1] == 1:
        risk_factors.append('Male')
        risk_score += 0.1
    
    # Chest pain
    if features[2] >= 2:  # Moderate to severe chest pain
        risk_factors.append('Moderate/Severe chest pain')
        risk_score += 0.3
    
    # Blood pressure
    if features[3] > 130:  # High blood pressure
        risk_factors.append(f'High blood pressure ({features[3]} mmHg)')
        risk_score += 0.2
    
    # Cholesterol
    if features[4] > 200:  # High cholesterol
        risk_factors.append(f'High cholesterol ({features[4]} mg/dL)')
        risk_score += 0.2
    
    # Fasting blood sugar
    if features[5] == 1:
        risk_factors.append('High fasting blood sugar')
        risk_score += 0.1
    
    # Exercise induced angina
    if features[8] == 1:
        risk_factors.append('Exercise induced angina')
        risk_score += 0.3
    
    # ST depression
    if features[9] > 1.0:  # Significant ST depression
        risk_factors.append(f'Significant ST depression ({features[9]} mm)')
        risk_score += 0.3
    
    # Number of major vessels
    if features[11] > 0:
        risk_factors.append(f'{int(features[11])} major vessels with reduced blood flow')
        risk_score += features[11] * 0.1
    
    # Thalassemia
    if features[12] == 3:  # Reversible defect
        risk_factors.append('Reversible defect (possible reduced blood flow)')
        risk_score += 0.3
    
    return risk_factors, min(1.0, risk_score)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not models:
            return jsonify({'error': 'No models available for prediction'}), 500
            
        # Get data from form
        data = request.json
        
        # Prepare features in correct order
        features = []
        for feature in FEATURE_NAMES:
            value = data.get(feature, '0')
            try:
                features.append(float(value) if value != '' else 0.0)
            except (ValueError, TypeError):
                print(f"Warning: Invalid value for {feature}: {value}, using 0")
                features.append(0.0)
        
        # Scale features
        features_array = np.array(features).reshape(1, -1)
        if scaler is not None:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Get risk factors and score
        risk_factors, risk_score = calculate_risk_factors(features)
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in models.items():
            try:
                # Handle models that might not have predict_proba
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0][1]
                else:
                    # For models without predict_proba, use decision function or predict
                    if hasattr(model, 'decision_function'):
                        decision = model.decision_function(features_scaled)[0]
                        proba = 1 / (1 + np.exp(-decision))  # Sigmoid function
                    else:
                        pred = model.predict(features_scaled)[0]
                        proba = 1.0 if pred == 1 else 0.0
                
                predictions[model_name] = float(proba)
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                # If prediction fails, use the average of other models
                if predictions:
                    predictions[model_name] = sum(predictions.values()) / len(predictions)
                else:
                    predictions[model_name] = 0.5  # Default to neutral probability
        
        if not predictions:
            return jsonify({'error': 'No models made successful predictions'}), 500
        
        # Calculate average probability
        avg_probability = sum(predictions.values()) / len(predictions)
        
        # Determine final prediction (threshold = 0.4 for high sensitivity)
        final_prediction = 1 if (avg_probability > 0.4 or risk_score > 0.5) else 0
        
        # Prepare response
        response = {
            'prediction': int(final_prediction),
            'probability': float(avg_probability),
            'risk_score': float(risk_score),
            'risk_factors': risk_factors,
            'model_predictions': predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Suppress Flask development server warning
    import warnings
    from flask import cli
    
    # Disable Flask dev server warning
    cli.show_server_banner = lambda *args: None
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Suppress Flask logging
    logging.getLogger('werkzeug').disabled = True
    
    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Custom startup message
    print("\n" + "="*60)
    print("Heart Disease Prediction App")
    print("="*60)
    print("Server running at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
