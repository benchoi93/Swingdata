import duckdb
import os
import time

INPUT_DIR = 'D:/SwingData/data_parquet'
OUTPUT_DIR = 'D:/SwingData/data_hive'

def reorganize():
    # Check input
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    con = duckdb.connect(database=':memory:')
    
    print(f"Source: {INPUT_DIR}")
    print(f"Destination: {OUTPUT_DIR}")
    print("Starting reorganization implementation... (COPY with PARTITION_BY)")
    
    try:
        # Check if we have files
        count = con.execute(f"SELECT count(*) FROM read_parquet('{INPUT_DIR}/*.parquet')").fetchone()[0]
        print(f"Found {count:,} records to reorganize.")
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Memory Optimization Settings
    con.execute("PRAGMA memory_limit='20GB'") # Leave room for OS
    con.execute("PRAGMA threads=4") # Reduce thread contention
    con.execute("SET preserve_insertion_order=false") # Reduce memory usage
    
    start_time = time.time()
    
    # Spatio-Temporal Partitioning Strategy:
    # 1. Year/Month (Time)
    # 2. Grid Lat/Lon (Space) - 0.1 degree resolution (~11km)
    
    # Processing Year by Year to prevent OOM
    years = [2022, 2023, 2024]
    
    for process_year in years:
        print(f"\nProcessing Year: {process_year}...")
        
        # Note: We filter by year in the WHERE clause
        query = f"""
        COPY (
            SELECT 
                *,
                year(start_timestamp) as year,
                month(start_timestamp) as month,
                CAST(FLOOR(list_extract(list_extract(path, 1), 2) * 10) AS INTEGER) as grid_lat,
                CAST(FLOOR(list_extract(list_extract(path, 1), 3) * 10) AS INTEGER) as grid_lon
            FROM read_parquet('{INPUT_DIR}/*.parquet')
            WHERE path IS NOT NULL 
              AND len(path) > 0
              AND year(start_timestamp) = {process_year}
        ) TO '{OUTPUT_DIR}' (FORMAT PARQUET, PARTITION_BY (year, month, grid_lat, grid_lon), OVERWRITE_OR_IGNORE 1);
        """
        
        try:
            iter_start = time.time()
            con.execute(query)
            print(f"Completed {process_year} in {time.time() - iter_start:.2f} seconds.")
        except Exception as e:
            print(f"Error processing {process_year}: {e}")

    duration = time.time() - start_time
    print(f"\nSUCCESS: Data reorganization completed in {duration:.2f} seconds.")

if __name__ == "__main__":
    reorganize()
