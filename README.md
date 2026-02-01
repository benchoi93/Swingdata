# Swing Scooter Data Visualizer

A data visualization tool for Swing scooter trajectories, built with Flask, DuckDB, and Deck.gl. This tool efficiently handles large-scale dataset visualization by leveraging Parquet files and DuckDB for query performance.

## Features
- Efficient preprocessing of large CSV datasets into Parquet.
- Real-time visualization of scooter routes.
- Time-based filtering.
- Interactive map interface.

## Prerequisites
- Python 3.8+
- Recommended: High-performance storage (SSD) for the `D:/` drive (configured as default output).

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

**Note:** The application is currently configured to store and read processed Parquet files from **`D:/SwingData/data_parquet`**.
Please ensure you have a `D:` drive or modify `scripts/preprocess.py` and `app/main.py` variables `OUTPUT_DIR` and `DATA_DIR` respectively if you wish to use a different location.

## Usage

### 1. Data Preprocessing

Before visualizing, raw CSV data must be converted to Parquet format.
Place your raw CSV file (e.g., `2023_05_Swing_Routes.csv`) in the root directory.

Run the preprocessor:
```bash
python scripts/preprocess.py
```
This will generate chunked Parquet files in the configured output directory.

### 2. Running the Visualizer

You can use the provided batch file (Windows):
```cmd
run_visualizer.bat
```

Or run via Python:
```bash
python app/main.py
```

Server will start at `http://127.0.0.1:5000`.

## Project Structure

- `app/`: Flask application source code.
- `scripts/`: Data processing and utility scripts.
- `data_parquet/`: Directory for processed data (if local).
- `analyze_csv.py`: Utility to quickly inspect raw CSV structure.

## Dependencies

- Flask
- Pandas
- DuckDB
- PyArrow
