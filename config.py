import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Debugging: Print loaded environment variables
print("DB_NAME:", os.getenv('DB_NAME'))
print("DB_USER:", os.getenv('DB_USER'))
print("DB_PASSWORD:", os.getenv('DB_PASSWORD'))
print("DB_HOST:", os.getenv('DB_HOST'))
print("API_KEY:", os.getenv('API_KEY'))

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'crypto_trading'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'host': os.getenv('DB_HOST', 'localhost')
}

# API key
API_KEY = os.getenv('API_KEY')