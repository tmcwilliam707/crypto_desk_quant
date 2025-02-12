# data_collector.py
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import psycopg2
from config import DB_CONFIG, API_KEY

class CryptoDataCollector:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
    
    def get_current_price(self, coin_id='bitcoin'):
        """Get current price from CoinGecko"""
        endpoint = f"{self.base_url}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true',
            'precision': '2'
        }
        headers = {
            'x-cg-demo-api-key': API_KEY  # Include the API key in the request header
        }
        
        print(f"Using API Key: {API_KEY}")  # Debugging: Print the API key
        
        try:
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return {
                'price': data[coin_id]['usd'],
                'volume_24h': data[coin_id].get('usd_24h_vol', 0.0),  # Handle missing field
                'change_24h': data[coin_id].get('usd_24h_change', 0.0),  # Handle missing field
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error collecting data: {e}")
            return None

    def get_historical_price(self, coin_id='bitcoin', date='01-01-2022'):
        """Get historical price from CoinGecko"""
        endpoint = f"{self.base_url}/coins/{coin_id}/history"
        params = {
            'date': date,
            'localization': 'false'
        }
        headers = {
            'x-cg-demo-api-key': API_KEY  # Include the API key in the request header
        }
        
        print(f"Using API Key: {API_KEY}")  # Debugging: Print the API key
        
        try:
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return {
                'price': data['market_data']['current_price']['usd'],
                'volume_24h': data['market_data'].get('total_volume', {}).get('usd', 0.0),  # Handle missing field
                'change_24h': data['market_data'].get('price_change_percentage_24h', 0.0),  # Handle missing field
                'timestamp': datetime.strptime(date, '%d-%m-%Y')
            }
        except Exception as e:
            print(f"Error collecting data: {e}")
            return None

    def store_price_data(self, data):
        """Store price data in PostgreSQL"""
        if not data:
            return
        
        query = """
        INSERT INTO price_data (timestamp, price, volume_24h, change_24h)
        VALUES (%s, %s, %s, %s)
        """
        
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute(query, (
                data['timestamp'],
                data['price'],
                data['volume_24h'],
                data['change_24h']
            ))
            conn.commit()
        except Exception as e:
            print(f"Error storing data: {e}")
        finally:
            if conn:
                conn.close()