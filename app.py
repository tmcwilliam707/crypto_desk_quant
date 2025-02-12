import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import psycopg2
from datetime import datetime
import numpy as np
from config import DB_CONFIG
import xgboost as xgb
from idealized_example import calculate_features, train_model, backtest_strategy, engine

class CryptoAnalytics:
    @staticmethod
    def calculate_features(df):
        """Calculate technical indicators"""
        df = df.copy()
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=30).std()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['ema_20'] = df['price'].ewm(span=20).mean()
        return df
    
    @staticmethod
    def train_model(features, target):
        """Train XGBoost model"""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1
        )
        model.fit(features, target)
        return model
    
    @staticmethod
    def backtest_strategy(df, model, feature_columns):
        """Backtest trading strategy"""
        df['prediction'] = model.predict(df[feature_columns])
        df['position'] = np.where(df['prediction'] > df['price'] * 1.001, 1,
                                np.where(df['prediction'] < df['price'] * 0.999, -1, 0))
        
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        sharpe_ratio = np.sqrt(252) * (df['strategy_returns'].mean() / 
                                      df['strategy_returns'].std())
        
        df['cumulative_max'] = df['cumulative_returns'].cummax()
        df['drawdown'] = (df['cumulative_returns'] - df['cumulative_max']) / \
                         df['cumulative_max']
        max_drawdown = df['drawdown'].min()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': df['cumulative_returns'].iloc[-1] - 1
        }

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Crypto Trading Dashboard", className="text-center my-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='price-chart', className='graph-container'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='technical-indicators', className='graph-container'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='prediction-chart', className='graph-container'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='metrics', className="p-3 bg-dark text-white"), width=12)
    ]),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every minute
        n_intervals=0
    )
], fluid=True)

# Callback to update charts
@app.callback(
    [Output('price-chart', 'figure'),
     Output('technical-indicators', 'figure'),
     Output('prediction-chart', 'figure'),
     Output('metrics', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_charts(n):
    try:
        # Load data from PostgreSQL
        df = pd.read_sql("SELECT * FROM price_data ORDER BY timestamp", engine)
        
        # Calculate features
        analytics = CryptoAnalytics()
        df = analytics.calculate_features(df)
        
        # Prepare data for model
        feature_columns = ['returns', 'volatility', 'sma_20', 'ema_20']
        features = df[feature_columns].dropna()
        target = df['price'].shift(-1).dropna()
        features, target = features.align(target, join='inner', axis=0)
        
        # Train model and make predictions
        model = analytics.train_model(features, target)
        df['prediction'] = model.predict(df[feature_columns].fillna(0))
        
        # Create price chart
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], 
                                     mode='lines', name='Price'))
        price_fig.update_layout(title='Bitcoin Price', template='plotly_dark')
        
        # Create technical indicators chart
        tech_fig = go.Figure()
        tech_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], 
                                    mode='lines', name='SMA 20'))
        tech_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], 
                                    mode='lines', name='EMA 20'))
        tech_fig.update_layout(title='Technical Indicators', template='plotly_dark')
        
        # Create prediction chart
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], 
                                    mode='lines', name='Price'))
        pred_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['prediction'], 
                                    mode='lines', name='Prediction'))
        pred_fig.update_layout(title='Price vs Prediction', template='plotly_dark')
        
        # Calculate metrics
        backtest_results = analytics.backtest_strategy(df, model, feature_columns)
        
        metrics = html.Div([
            html.H4("Trading Metrics"),
            html.P(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}"),
            html.P(f"Max Drawdown: {backtest_results['max_drawdown']:.4f}"),
            html.P(f"Total Return: {backtest_results['total_return']:.4f}")
        ])
        
        return price_fig, tech_fig, pred_fig, metrics
        
    except Exception as e:
        print(f"Error updating charts: {e}")
        return {}, {}, {}, "Error updating dashboard"

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)