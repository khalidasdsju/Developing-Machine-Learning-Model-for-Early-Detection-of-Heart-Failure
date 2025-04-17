"""
API module for the Heart Failure Detection project.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

from config.config import BEST_MODEL_PATH, API_HOST, API_PORT, API_DEBUG
from src.data_processing.preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load model and preprocessor
model = None
preprocessor = None

def load_model_and_preprocessor():
    """
    Load the model and preprocessor.
    
    Returns:
        tuple: Tuple containing (model, preprocessor)
    """
    try:
        # Load model
        model = joblib.load(BEST_MODEL_PATH)
        logger.info(f"Model loaded from {BEST_MODEL_PATH}")
        
        # Load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_transformers()
        
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {str(e)}")
        return None, None

@app.route('/')
def home():
    """
    Render the home page.
    
    Returns:
        str: Rendered HTML template
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions based on input data.
    
    Returns:
        dict: JSON response with prediction results
    """
    try:
        # Get input data
        data = request.json
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess input data
        preprocessed_data = preprocessor.preprocess_new_data(input_df)
        
        # Make prediction
        prediction = model.predict(preprocessed_data)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(preprocessed_data)[0][1]
        else:
            probability = None
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': float(probability) if probability is not None else None,
            'message': 'Heart failure detected' if prediction == 1 else 'No heart failure detected'
        }
        
        logger.info(f"Prediction made: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Error making prediction'
        }), 500

@app.route('/health')
def health():
    """
    Health check endpoint.
    
    Returns:
        dict: JSON response with health status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

def start_api():
    """
    Start the Flask API.
    """
    global model, preprocessor
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    
    # Start Flask app
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)

if __name__ == '__main__':
    start_api()
