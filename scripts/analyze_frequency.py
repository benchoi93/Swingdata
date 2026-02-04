import duckdb
import os
import numpy as np

DATA_DIR = 'D:/SwingData/data_parquet'

def analyze():
    print(f"Analyzing data frequency in {DATA_DIR}...")
    try:
        con = duckdb.connect(database=':memory:')
        path = os.path.join(DATA_DIR, "*.parquet").replace("\\", "/")
        
        # Get a sample of paths
        # We need to pull the list of lists. DuckDB returns this as a list of structs or lists
        query = f"SELECT path FROM read_parquet('{path}') LIMIT 100"
        results = con.execute(query).fetchall()
        
        intervals = []
        
        for row in results:
            # row[0] is the path list
            path_data = row[0]
            if not path_data or len(path_data) < 2:
                continue
                
            # path_data is likely a list of [ts, lat, lon]
            # Verify structure (DuckDB might return numpy arrays or python lists)
            
            timestamps = [p[0] for p in path_data]
            
            # Calculate diffs
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                # Filter out likely gaps/pauses (e.g., > 60 seconds) to find the "active" capture rate
                if diff > 0 and diff < 60: 
                    intervals.append(diff)
                    
        if not intervals:
            print("No valid intervals found.")
            return

        intervals = np.array(intervals)
        
        print(f"\n--- Analysis Base on {len(intervals)} segments from 100 trips ---")
        print(f"Mean Interval: {np.mean(intervals):.4f} seconds")
        print(f"Median Interval: {np.median(intervals):.4f} seconds")
        print(f"Min Interval: {np.min(intervals):.4f} seconds")
        print(f"Max Interval: {np.max(intervals):.4f} seconds")
        
        # Histogram-like output
        unique, counts = np.unique(np.round(intervals, 1), return_counts=True)
        print("\nCommon Intervals (rounded to 0.1s):")
        # Sort by count desc
        sorted_indices = np.argsort(-counts)
        for i in sorted_indices[:5]:
            print(f"{unique[i]}s: {counts[i]} times")

    except Exception as e:
        print(f"Analysis Failed: {e}")

if __name__ == "__main__":
    analyze()
