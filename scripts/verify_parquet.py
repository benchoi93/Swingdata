import duckdb
import os

DATA_DIR = 'D:/SwingData/data_parquet'

def verify():
    print(f"Checking data in {DATA_DIR}...")
    try:
        # Connect to DuckDB
        con = duckdb.connect(database=':memory:')
        
        # Query the files
        path = os.path.join(DATA_DIR, "*.parquet").replace("\\", "/")
        print(f"Querying {path}...")
        
        count = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        print(f"Total rows so far: {count}")
        
        print("Sample row:")
        sample = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 1").fetchdf()
        print(sample)
        
        print("\nVerification SUCCESS")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")

if __name__ == "__main__":
    verify()
