"""
Project configuration — all paths and constants in one place.
"""
from pathlib import Path

# --- Directories ---
PROJECT_ROOT = Path("C:/Users/chois/Gitsrcs/Swingdata")
DATA_DIR = PROJECT_ROOT / "data_parquet"
RAW_DIR = PROJECT_ROOT  # CSVs are in project root
SRC_DIR = PROJECT_ROOT / "src"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Raw data files ---
ROUTES_CSV = RAW_DIR / "2023_05_Swing_Routes.csv"       # ~2.83M rows, GPS trajectories
SCOOTER_CSV = RAW_DIR / "2023_Swing_Scooter.csv"        # ~20M rows, trip-level

# --- Processed data outputs ---
CLEANED_PARQUET = DATA_DIR / "cleaned"
TRIP_INDICATORS_PARQUET = DATA_DIR / "trip_indicators.parquet"
SEGMENT_INDICATORS_PARQUET = DATA_DIR / "segment_indicators.parquet"
USER_INDICATORS_PARQUET = DATA_DIR / "user_indicators.parquet"
OSM_NETWORKS_DIR = DATA_DIR / "osm_networks"
MODELING_DIR = DATA_DIR / "modeling"

# --- Constants ---
SPEED_LIMIT_KR = 25        # km/h — Korean regulatory limit for e-scooters
SPEED_THRESHOLDS = [15, 20, 25, 30]  # km/h — for sensitivity analysis
HARSH_ACCEL_THRESHOLD = 2.0  # m/s^2 — for high-freq data (NOT suitable for 10s intervals)
HARSH_ACCEL_THRESHOLD_10S = 0.5  # m/s^2 — calibrated for ~10s speed intervals
MIN_TRIP_POINTS = 5         # minimum GPS points for a valid trip
MIN_TRIP_DISTANCE = 100     # meters — minimum trip distance
MIN_TRIP_DURATION = 60      # seconds — minimum trip duration
MAX_PLAUSIBLE_SPEED = 50    # km/h — filter GPS errors above this
RANDOM_SEED = 42
CHUNK_SIZE = 50_000         # rows per chunk for CSV reading

# --- GPS quality ---
MAX_GPS_GAP = 120           # seconds — max acceptable gap between GPS points

# --- Models to exclude (GPS errors) ---
EXCLUDE_MODELS = ["I9"]     # I9 has implausible speeds (mean max ~83 km/h)

# --- Spatial analysis ---
H3_RESOLUTION = 8           # ~250m hexagons

# --- Figure settings ---
FIG_DPI = 300
FIG_FORMAT = "pdf"          # primary format for publication
FIG_FORMAT_SECONDARY = "png"  # for reports and previews
