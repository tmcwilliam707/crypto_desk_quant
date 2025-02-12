# analysis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import psycopg2
from config import DB_CONFIG

class CryptoAnalyzer:
    def __init__(self):
        self.data = None
        
    def load_data(self):
        """Load data from PostgreSQL"""
        query = "SELECT * FROM price_data ORDER BY timestamp"
        
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            self.data = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
        return self.data
    
    def calculate_features(self):
        """Calculate technical indicators"""
        if self.data is None:
            return None
            
        df = self.data.copy()
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        
        # Calculate moving averages
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate momentum
        df['momentum'] = df['price'].pct_change(periods=10)
        
        return df
        
    def calculate_risk_metrics(self):
        """Calculate basic risk metrics"""
        if self.data is None:
            return None
            
        returns = self.data['returns'].dropna()
        
        risk_metrics = {
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'var_95': returns.quantile(0.05),  # 95% VaR
            'max_drawdown': (self.data['price'] / self.data['price'].cummax() - 1).min(),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized Sharpe
        }
        
        return risk_metrics
