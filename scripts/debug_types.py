import duckdb
import os
import pandas as pd
import numpy as np

DATA_DIR = 'D:/SwingData/data_parquet'

def debug():
    con = duckdb.connect(database=':memory:')
    path = os.path.join(DATA_DIR, "*.parquet").replace("\\", "/")
    
    print("Fetching one row...")
    df = con.execute(f"SELECT path FROM read_parquet('{path}') LIMIT 1").fetchdf()
    
    val = df['path'].iloc[0]
    print(f"Type of path column content: {type(val)}")
    
    if hasattr(val, 'dtype'):
        print(f"Dtype: {val.dtype}")
        
    if isinstance(val, np.ndarray):
        print("It is a numpy array.")
        print(f"Shape: {val.shape}")
        # Test tolist
        print("Testing tolist()...")
        l = val.tolist()
        print(f"Result type: {type(l)}")
        print(f"Result first element type: {type(l[0]) if len(l)>0 else 'empty'}")
        
    print("Content preview:", val)

if __name__ == "__main__":
    debug()
