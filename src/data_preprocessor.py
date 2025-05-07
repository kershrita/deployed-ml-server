import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Optional

class DataPreprocessor:
    """Handles data preprocessing tasks including cleaning, encoding, and scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = ['gender', 'city']
        self.numerical_columns = ['age', 'income', 'days_on_platform']
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the dataset by handling missing values and invalid entries."""
        df = df.copy()
        
        # Replace empty strings with NaN
        df.replace('', np.nan, inplace=True)
        
        # Fill missing numerical values with median
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
                
        # Fill missing categorical values with mode
        for col in self.categorical_columns:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encodes categorical variables using LabelEncoder."""
        df = df.copy()
        
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col])
                    else:
                        raise ValueError(f"No encoder found for column {col}")
                        
        return df
    
    def scale_numerical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scales numerical features using StandardScaler."""
        df = df.copy()
        
        if fit:
            df[self.numerical_columns] = self.scaler.fit_transform(df[self.numerical_columns])
        else:
            df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
            
        return df
    
    def preprocess(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Applies full preprocessing pipeline."""
        df = self.clean_data(df)
        df = self.encode_categorical(df, fit=fit)
        df = self.scale_numerical(df, fit=fit)
        return df
    
    def preprocess_single(self, data: Dict) -> np.ndarray:
        """Preprocesses a single data point for prediction."""
        df = pd.DataFrame([data])
        df = self.preprocess(df, fit=False)
        return df[self.numerical_columns + self.categorical_columns].values
