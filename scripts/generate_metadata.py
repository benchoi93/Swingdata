import duckdb
import json
import os
import time

DATA_DIR = 'D:/SwingData/data_hive'
METADATA_FILE = os.path.join(DATA_DIR, 'metadata.json')

def generate_metadata():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    print("Generating metadata from Hive partitions...")
    start_time = time.time()
    
    con = duckdb.connect(database=':memory:')
    
    # 1. Get Totals
    # Using the hive structure, we can just read the parquet files. 
    # hive_partitioning=1 lets us query year/month columns if needed, but for min/max we just need the timestamps.
    
    # NOTE: Reading 30M rows might still take 10-20s. 
    # Since this is an offline script, we can afford it.
    
    print("Calculating statistics (count, min, max)...")
    path = os.path.join(DATA_DIR, "**", "*.parquet").replace("\\", "/")
    
    query = f"""
        SELECT 
            count(*) as total_count,
            min(start_timestamp) as min_ts,
            max(end_timestamp) as max_ts
        FROM read_parquet('{path}', hive_partitioning=1)
    """
    
    stats = con.execute(query).fetchone()
    total_count = stats[0]
    min_ts = stats[1]
    max_ts = stats[2]
    
    print(f"Total Records: {total_count}")
    print(f"Time Range: {min_ts} to {max_ts}")
    
    metadata = {
        "total_records": total_count,
        "start_date": min_ts.isoformat() if min_ts else None,
        "end_date": max_ts.isoformat() if max_ts else None,
        "generated_at": time.time()
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Metadata saved to {METADATA_FILE}")
    print(f"Duration: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    generate_metadata()
