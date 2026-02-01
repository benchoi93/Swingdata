from flask import Flask, jsonify, request, render_template
import duckdb
import os
import glob

app = Flask(__name__)

# Configuration
# Pointing to the D: drive output directory
DATA_DIR = 'D:/SwingData/data_parquet'
DB_CONNECTION = None

def get_db_connection():
    """Establishes or returns a DuckDB connection to the Parquet files."""
    global DB_CONNECTION
    if DB_CONNECTION is None:
        # We can query parquet files directly using glob syntax
        # Check if we have any files first
        parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
        if not parquet_files:
            print("No parquet files found in", DATA_DIR)
            return None
        
        # Connect to in-memory DuckDB, but we will query the files directly
        DB_CONNECTION = duckdb.connect(database=':memory:')
        
        # Create a view for easier access
        # This wildcard works in DuckDB to read all matching files
        data_path = os.path.join(DATA_DIR, "*.parquet").replace("\\", "/")
        query = f"CREATE OR REPLACE VIEW scooter_data AS SELECT * FROM read_parquet('{data_path}');"
        print(f"Creating view with query: {query}")
        DB_CONNECTION.execute(query)
        
    return DB_CONNECTION

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
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
    """Return points for visualization, optionally filtered by time."""
    con = get_db_connection()
    if not con:
        return jsonify({"error": "No data available"}), 503
    
    # Get parameters
    limit = request.args.get('limit', 5000) # Increased default sample
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    
    try:
        query = "SELECT route_id, start_timestamp, end_timestamp, path FROM scooter_data"
        conditions = []
        
        if start_str:
            conditions.append(f"start_timestamp >= '{start_str}'")
        if end_str:
            conditions.append(f"end_timestamp <= '{end_str}'")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        # Hard cap to prevent browser crash
        query += f" LIMIT {limit}"
        
        print(f"Executing: {query}")
        df = con.execute(query).fetchdf()
        
        # Recursive fix from before
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
