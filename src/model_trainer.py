from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.data_preprocessor import DataPreprocessor
import pandas as pd
import pickle
from typing import Tuple

class ModelTrainer:
    """Trains and saves a classification model."""
    
    def __init__(self, model_path: str):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.preprocessor = DataPreprocessor()
        self.model_path = model_path
        
    def train(self, data_path: str) -> None:
        """Trains the model on the provided dataset."""
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess data
        df_processed = self.preprocessor.preprocess(df, fit=True)
        
        # Split features and target
        X = df_processed[self.preprocessor.numerical_columns + self.preprocessor.categorical_columns]
        y = df['purchases']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessor': self.preprocessor
            }, f)
            
        print(f"Model trained and saved to {self.model_path}")
