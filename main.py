import time
import psycopg2
from config import DB_CONFIG
from data_collector import CryptoDataCollector
from analysis import CryptoAnalyzer

def setup_database():
    """Create necessary database tables"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS price_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        price FLOAT NOT NULL,
        volume_24h FLOAT,
        change_24h FLOAT
    );
    """
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
        print("Database setup completed successfully")
    except Exception as e:
        print(f"Error setting up database: {e}")
    finally:
        if conn:
            conn.close()

def main():
    # Setup database
    setup_database()
    
    # Initialize collector
    collector = CryptoDataCollector()
    
    # Collect and store data
    print("Collecting data...")
    for i in range(10):  # Collect 10 data points
        print(f"Collecting data point {i+1}/10")
        data = collector.get_current_price()
        if data:
            collector.store_price_data(data)
        time.sleep(60)  # Wait 1 minute between collections
    
    # Analyze data
    print("\nAnalyzing data...")
    analyzer = CryptoAnalyzer()
    data = analyzer.load_data()
    
    if data is not None:
        features = analyzer.calculate_features()
        risk_metrics = analyzer.calculate_risk_metrics()
        
        print("\nRisk Metrics:")
        for metric, value in risk_metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()