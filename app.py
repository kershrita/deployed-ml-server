from flask import Flask, request, jsonify
from src.data_preprocessor import DataPreprocessor
import pickle
import os
from dotenv import load_dotenv
from typing import Dict, Any

class MLServer:
    """Flask-based server for serving ML model predictions."""
    
    def __init__(self, model_path: str, api_key: str = None):
        self.app = Flask(__name__)
        self.model_path = model_path
        self.api_key = api_key
        self.model = None
        self.preprocessor = None
        self._load_model()
        self._register_routes()
        
    def _load_model(self) -> None:
        """Loads the trained model and preprocessor."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.preprocessor = model_data['preprocessor']
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _register_routes(self) -> None:
        """Registers API routes."""
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Handles prediction requests."""
            # Check API key
            if self.api_key:
                provided_key = request.headers.get('X-API-KEY')
                if provided_key != self.api_key:
                    return jsonify({'error': 'Invalid API key'}), 401
            
            # Get input data
            try:
                data = request.get_json()
                required_fields = ['age', 'gender', 'income', 'days_on_platform', 'city']
                if not all(field in data for field in required_fields):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Preprocess and predict
                processed_data = self.preprocessor.preprocess_single(data)
                prediction = self.model.predict(processed_data)[0]
                return jsonify({'prediction': int(prediction)})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000) -> None:
        """Runs the Flask server."""
        self.app.run(host=host, port=port)

# Create MLServer instance and expose Flask app
load_dotenv()
api_key = os.getenv('API_KEY')
server = MLServer(model_path='models/classification_model.pkl', api_key=api_key)
app = server.app

if __name__ == '__main__':
    server.run()
