import pandas as pd
import numpy as np
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import plotly.graph_objects as go
from dask.distributed import Client
from config import DB_CONFIG
import requests
import time
from sqlalchemy import create_engine

# Create SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['dbname']}")

# 1. Data Ingestion from CoinGecko API
def fetch_data_from_api(coin_id='bitcoin'):
    endpoint = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': coin_id,
        'vs_currencies': 'usd',
        'include_market_cap': 'true',
        'include_24hr_vol': 'true',
        'include_24hr_change': 'true',
        'include_last_updated_at': 'true',
        'precision': '2'
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            'timestamp': pd.Timestamp.now(),
            'price': data[coin_id]['usd'],
            'volume_24h': data[coin_id].get('usd_24h_vol', 0.0),
            'change_24h': data[coin_id].get('usd_24h_change', 0.0)
        }
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return None

# 2. Feature Engineering
def calculate_features(df):
    # Technical indicators
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=30).std()
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['ema_20'] = df['price'].ewm(span=20).mean()
    return df

# 3. Model Training
def train_model(features, target):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1
    )
    model.fit(features, target)
    return model

# 4. Backtesting
def backtest_strategy(df, model, feature_columns, initial_capital=100000):
    df['prediction'] = model.predict(df[feature_columns])
    df['position'] = np.where(df['prediction'] > df['price'] * 1.001, 1, 
                            np.where(df['prediction'] < df['price'] * 0.999, -1, 0))
    
    # Calculate returns
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    
    # Calculate metrics
    sharpe_ratio = np.sqrt(252) * (df['strategy_returns'].mean() / 
                                  df['strategy_returns'].std())
    
    # Calculate drawdown
    df['cumulative_max'] = df['cumulative_returns'].cummax()
    df['drawdown'] = (df['cumulative_returns'] - df['cumulative_max']) / \
                     df['cumulative_max']
    max_drawdown = df['drawdown'].min()
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_return': df['cumulative_returns'].iloc[-1] - 1
    }

# 5. Risk Metrics
def calculate_risk_metrics(returns, confidence_level=0.95):
    # Value at Risk (VaR)
    var = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Conditional Value at Risk (CVaR)
    cvar = returns[returns <= var].mean()
    
    return {
        'VaR': var,
        'CVaR': cvar
    }

# Main workflow
def main():
    # Setup Dask client
    client = Client()

    # Create table if it does not exist
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS price_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            price FLOAT NOT NULL,
            volume_24h FLOAT,
            change_24h FLOAT
        );
        """
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating table: {e}")
        return

    # Periodically fetch data from API
    while True:
        # Fetch data from API
        data = fetch_data_from_api()
        
        if data:
            # Store in PostgreSQL
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor()
                query = """
                INSERT INTO price_data (timestamp, price, volume_24h, change_24h)
                VALUES (%s, %s, %s, %s)
                """
                cur.execute(query, (
                    data['timestamp'],
                    data['price'],
                    data['volume_24h'],
                    data['change_24h']
                ))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Error storing data: {e}")
            
            # Load data from PostgreSQL
            try:
                df = pd.read_sql("SELECT * FROM price_data ORDER BY timestamp", engine)
            except Exception as e:
                print(f"Error loading data: {e}")
                continue
            
            # Update models and risk metrics
            df = calculate_features(df)
            
            # Train model
            feature_columns = ['returns', 'volatility', 'sma_20', 'ema_20']
            features = df[feature_columns].dropna()
            target = df['price'].shift(-1).dropna()
            features, target = features.align(target, join='inner', axis=0)  # Align indices
            model = train_model(features, target)
            
            # Backtest strategy
            backtest_results = backtest_strategy(df, model, feature_columns)
            print("\nBacktest Results:")
            for metric, value in backtest_results.items():
                print(f"{metric}: {value:.4f}")
            
            # Update risk metrics
            risk_metrics = calculate_risk_metrics(df['returns'].dropna())
            print("\nRisk Metrics:")
            for metric, value in risk_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Monitor for arbitrage opportunities
            # ... (arbitrage detection logic)
        
        # Wait for a minute before fetching new data
        time.sleep(60)

if __name__ == "__main__":
    main()