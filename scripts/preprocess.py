import pandas as pd
import ast
import os
import glob
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = 'D:/SwingData/raw'
OUTPUT_DIR = 'D:/SwingData/data_parquet'
CHUNK_SIZE = 10000

def parse_route(route_str):
    try:
        # route_str is a string representation of a list of lists
        # data format: [['2023/05/01', '00:00:17.660', lat, lng], ...]
        raw_points = ast.literal_eval(route_str)
        
        parsed_points = []
        for p in raw_points:
            if len(p) >= 4:
                date_str = p[0]
                time_str = p[1]
                lat = p[2]
                lon = p[3]
                
                # Combine date and time
                dt_str = f"{date_str} {time_str}"
                try:
                    dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S.%f")
                    ts = dt.timestamp()
                except ValueError:
                    # Fallback for missing milliseconds or other formats
                    try:
                        dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
                        ts = dt.timestamp()
                    except:
                        continue
                
                parsed_points.append([ts, lat, lon])
        return parsed_points
    except:
        return []

def preprocess():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Get list of all CSV files
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    print(f"Found {len(csv_files)} CSV files in {INPUT_DIR}")
    
    for file_index, input_file in enumerate(csv_files):
        filename = os.path.basename(input_file)
        file_base = os.path.splitext(filename)[0]
        print(f"[{file_index+1}/{len(csv_files)}] Processing {filename}...")
        
        try:
            # Process in chunks
            chunk_iter = pd.read_csv(input_file, chunksize=CHUNK_SIZE)
            
            for i, chunk in enumerate(chunk_iter):
                print(f"  Processing chunk {i+1}...", end='\r')
                
                # Apply parsing
                chunk['path'] = chunk['routes'].apply(parse_route)
                
                # Select relevant columns
                # Filter out rows where path is empty if desired
                chunk = chunk[chunk['path'].map(len) > 0]
                
                # Create timestamps
                chunk['start_timestamp'] = pd.to_datetime(chunk['start_date'] + ' ' + chunk['start_time'])
                chunk['end_timestamp'] = pd.to_datetime(chunk['end_date'] + ' ' + chunk['end_time'])
                
                keep_cols = [
                    'route_id', 'user_id', 'model', 'travel_time', 'distance', 
                    'start_timestamp', 'end_timestamp', 'path'
                ]
                
                # Ensure we only check columns that exist (in case of schema drift)
                existing_cols = [c for c in keep_cols if c in chunk.columns]
                cleaned_chunk = chunk[existing_cols]
                
                # Convert to Table
                table = pa.Table.from_pandas(cleaned_chunk)
                
                # Write to Parquet with unique name including source filename
                output_file = os.path.join(OUTPUT_DIR, f"{file_base}_chunk_{i:04d}.parquet")
                pq.write_table(table, output_file)
            
            print(f"\nCompleted {filename}")
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

if __name__ == '__main__':
    preprocess()
