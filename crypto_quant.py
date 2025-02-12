"""
Crypto Quantitative Trading System

This script demonstrates a simplified version of a quantitative crypto trading system. It fetches real-time market data from Binance, executes trades,
monitors risk exposure, and provides post-trade performance analysis.

Best Practices in Financial Trading:
1. **Risk Management:** Always ensure a well-defined risk strategy, including stop-loss limits, portfolio diversification, and position sizing.
2. **Liquidity Consideration:** Trade assets with high liquidity to avoid slippage and ensure smooth order execution.
3. **Backtesting & Strategy Optimization:** Before deploying a trading strategy, conduct thorough backtesting on historical data.
4. **Market Data Reliability:** Always validate market data sources and implement fail-safes to prevent incorrect trade execution.
5. **Regulatory Compliance:** Follow legal regulations when trading crypto, especially regarding tax implications and reporting requirements.

"""

import numpy as np
import pandas as pd
import time
import requests

class CryptoQuantTrader:
    """
    A quantitative crypto trading system that:
    - Fetches real-time market data from Binance
    - Executes trades based on predefined rules
    - Monitors risk exposure
    - Conducts post-trade performance analysis
    """
    
    def __init__(self, initial_balance=100000):
        """Initialize the trading system with an initial balance."""
        self.balance = initial_balance
        self.positions = {}
        self.trade_log = []

    def fetch_market_data(self, symbol="BTCUSDT"):
        """Fetch the latest market price for a given symbol from Binance."""
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url).json()
        return float(response["price"])

    def execute_trade(self, symbol, size):
        """Execute a trade by purchasing a given size of a crypto asset."""
        price = self.fetch_market_data(symbol)
        cost = size * price
        
        # Ensure there is enough balance to execute the trade
        if cost <= self.balance:
            self.balance -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + size
            self.trade_log.append((symbol, size, price, time.time()))
            print(f"Trade Executed: {symbol} | Size: {size} | Price: {price}")
        else:
            print("Trade Failed: Insufficient Balance")

    def monitor_risk(self):
        """Monitor portfolio risk exposure by calculating the ratio of held assets to total portfolio value."""
        total_value = sum(size * self.fetch_market_data(symbol) for symbol, size in self.positions.items())
        risk_ratio = total_value / (self.balance + total_value)
        print(f"Current Risk Exposure: {risk_ratio:.2%}")
        return risk_ratio

    def post_trade_analysis(self):
        """Analyze trading performance by summarizing executed trades."""
        df = pd.DataFrame(self.trade_log, columns=["Symbol", "Size", "Price", "Timestamp"])
        print("Post-Trade Performance Analysis")
        print(df.describe())

# Example usage
print("Initializing Crypto Quant Trading System...")
trader = CryptoQuantTrader()
for _ in range(5):
    trader.execute_trade("BTCUSDT", np.random.uniform(0.01, 0.1))
    time.sleep(1)

trader.post_trade_analysis()
