from flask import Flask, jsonify, request, render_template
import duckdb
import os
import glob

app = Flask(__name__)

# Configuration
# Pointing to the new Hive-partitioned directory
DATA_DIR = 'D:/SwingData/data_hive'
DB_CONNECTION = None

def get_db_connection():
    """Establishes or returns a DuckDB connection to the Parquet files."""
    global DB_CONNECTION
    if DB_CONNECTION is None:
        # Check if we have any files first (recursive check)
        # In Hive structure: year=*/month=*/day=*/*.parquet
        if not os.path.exists(DATA_DIR):
             print("Data directory not found:", DATA_DIR)
             return None
             
        # Connect to in-memory DuckDB
        DB_CONNECTION = duckdb.connect(database=':memory:')
        
        # Create a view using Hive Partitioning
        # We point to the root directory and enable hive_partitioning
        # Note: glob pattern should match the files, typically **/*.parquet
        data_path = os.path.join(DATA_DIR, "**", "*.parquet").replace("\\", "/")
        
        # DuckDB auto-infers partitions from directory structure if hive_partitioning=1
        query = f"CREATE OR REPLACE VIEW scooter_data AS SELECT * FROM read_parquet('{data_path}', hive_partitioning=1);"
        print(f"Creating view with query: {query}")
        try:
            DB_CONNECTION.execute(query)
        except Exception as e:
            print(f"Failed to create view: {e}")
            return None
        
    return DB_CONNECTION

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    # Optimization: Read from metadata.json if available
    metadata_file = os.path.join(DATA_DIR, 'metadata.json')
    if os.path.exists(metadata_file):
        try:
            import json
            with open(metadata_file, 'r') as f:
                return jsonify(json.load(f))
        except Exception as e:
            print(f"Error reading metadata: {e}")
            
    con = get_db_connection()
    if not con:
        return jsonify({"error": "No data available yet"}), 503
        
    try:
        # Simple count query
        res_count = con.execute("SELECT count(*) FROM scooter_data").fetchone()
        count = res_count[0] if res_count else 0
        
        # Get date range
        min_max = con.execute("SELECT min(start_timestamp), max(end_timestamp) FROM scooter_data").fetchone()
        
        start_date = None
        end_date = None
        
        if min_max:
             if min_max[0]:
                 start_date = min_max[0].isoformat()
             if min_max[1]:
                 end_date = min_max[1].isoformat()
        
        return jsonify({
            "total_records": count,
            "start_date": start_date,
            "end_date": end_date
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/sample')
def sample():
    """Return points for visualization, optionally filtered by time and region."""
    con = get_db_connection()
    if not con:
        return jsonify({"error": "No data available"}), 503
    
    # Get parameters
    limit = request.args.get('limit', 5000)
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    
    # Bounding box params
    north = request.args.get('north')
    south = request.args.get('south')
    east = request.args.get('east')
    west = request.args.get('west')
    
    try:
        query = "SELECT route_id, start_timestamp, end_timestamp, path FROM scooter_data"
        conditions = []
        
        if start_str:
            conditions.append(f"start_timestamp >= '{start_str}'")
        if end_str:
            conditions.append(f"end_timestamp <= '{end_str}'")
            
        # Region filtering
        if north and south and east and west:
            # Cast to float
            n, s, e, w = float(north), float(south), float(east), float(west)
            
            # 1. Exact Spatial Filter (Points must start in bounds)
            conditions.append(f"path[1][2] BETWEEN {s} AND {n}")
            conditions.append(f"path[1][3] BETWEEN {w} AND {e}")
            
            # 2. Partition Pruning (Optimization)
            # We filter by the grid columns so DuckDB skips irrelevant folders
            min_grid_lat = int(s * 10) # FLOOR
            max_grid_lat = int(n * 10)
            min_grid_lon = int(w * 10)
            max_grid_lon = int(e * 10)
            
            conditions.append(f"grid_lat BETWEEN {min_grid_lat} AND {max_grid_lat}")
            conditions.append(f"grid_lon BETWEEN {min_grid_lon} AND {max_grid_lon}")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        # Hard cap to prevent browser crash
        query += f" LIMIT {limit}"
        
        print(f"Executing: {query}")
        df = con.execute(query).fetchdf()
        
        import numpy as np
        def to_list(x):
            if isinstance(x, np.ndarray):
                return [to_list(i) for i in x]
            elif isinstance(x, list):
                return [to_list(i) for i in x]
            return x
            
        df['path'] = df['path'].apply(to_list)
        
        result = df.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
