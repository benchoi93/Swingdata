"""
Tasks 2.1–2.3: Compute trip-level speed and safety indicators.

For each trip, parses the speeds column and computes:
  - Trip-level speed indicators: mean, max, P85, speed CV, speeding rate, speeding duration
  - Acceleration/deceleration: from consecutive speed differences, harsh events
  - Within-trip speed profile features: ramp-up duration, cruise speed/duration

Outputs:
  - data_parquet/trip_indicators.parquet
"""

import ast
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, SPEED_LIMIT_KR, HARSH_ACCEL_THRESHOLD,
    SPEED_THRESHOLDS, RANDOM_SEED,
)

# Assumed time interval between speed readings (seconds)
# From validation: GPS interval ~10s, but speeds count != GPS count
# The speeds are likely recorded at ~10s intervals
SPEED_INTERVAL_S = 10.0

# Chunk size for processing
CHUNK_SIZE = 100_000


def parse_speeds_fast(speeds_str: str) -> Optional[np.ndarray]:
    """Parse speeds string into numpy array.

    Args:
        speeds_str: Stringified nested list, e.g. '[[18, 16, 21]]'

    Returns:
        Numpy array of speeds in km/h, or None if parsing fails.
    """
    try:
        raw = ast.literal_eval(speeds_str)
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            return np.array(raw[0], dtype=np.float32)
        elif isinstance(raw, list):
            return np.array(raw, dtype=np.float32)
    except (ValueError, SyntaxError):
        pass
    return None


def compute_trip_indicators(speeds: np.ndarray) -> dict:
    """Compute all speed and safety indicators for a single trip.

    Args:
        speeds: Array of speed values in km/h.

    Returns:
        Dictionary of computed indicators.
    """
    n = len(speeds)
    if n == 0:
        return _empty_indicators()

    # --- Task 2.1: Trip-level speed indicators ---
    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    min_speed = float(np.min(speeds))
    p85_speed = float(np.percentile(speeds, 85))
    p95_speed = float(np.percentile(speeds, 95))
    speed_std = float(np.std(speeds))
    speed_cv = speed_std / mean_speed if mean_speed > 0 else 0.0

    # Speeding metrics for each threshold
    above_25 = speeds > SPEED_LIMIT_KR
    speeding_rate_25 = float(np.mean(above_25))  # fraction of points > 25
    speeding_count_25 = int(np.sum(above_25))
    speeding_duration_25 = speeding_count_25 * SPEED_INTERVAL_S  # time-weighted (seconds)

    # Speeding at other thresholds (for sensitivity analysis)
    speeding_rate_20 = float(np.mean(speeds > 20))
    speeding_rate_30 = float(np.mean(speeds > 30))
    speeding_rate_15 = float(np.mean(speeds > 15))

    # Max excess speed (how far above limit)
    max_excess_25 = max(0.0, float(max_speed - SPEED_LIMIT_KR))

    # --- Task 2.2: Acceleration/deceleration ---
    if n >= 2:
        # Speed differences (km/h per interval)
        speed_diffs = np.diff(speeds)  # km/h per ~10s interval

        # Convert to m/s^2: (km/h per 10s) * (1000/3600) / 10 = km/h per 10s * (1/36)
        accel_ms2 = speed_diffs * (1000.0 / 3600.0) / SPEED_INTERVAL_S

        mean_accel = float(np.mean(np.abs(accel_ms2)))
        max_accel = float(np.max(accel_ms2))
        max_decel = float(np.min(accel_ms2))  # most negative

        # Harsh events
        harsh_accel_count = int(np.sum(accel_ms2 > HARSH_ACCEL_THRESHOLD))
        harsh_decel_count = int(np.sum(accel_ms2 < -HARSH_ACCEL_THRESHOLD))
        harsh_event_count = harsh_accel_count + harsh_decel_count
        harsh_event_rate = harsh_event_count / (n - 1)
    else:
        mean_accel = 0.0
        max_accel = 0.0
        max_decel = 0.0
        harsh_accel_count = 0
        harsh_decel_count = 0
        harsh_event_count = 0
        harsh_event_rate = 0.0

    # --- Task 2.3: Within-trip speed profile features ---
    # Ramp-up: number of intervals from start until first reaching 80% of max speed
    threshold_80 = max_speed * 0.8
    ramp_up_idx = np.argmax(speeds >= threshold_80) if np.any(speeds >= threshold_80) else n
    ramp_up_duration = float(ramp_up_idx * SPEED_INTERVAL_S)

    # Cruise detection: periods where speed is within +/- 3 km/h of mean
    cruise_mask = np.abs(speeds - mean_speed) <= 3.0
    cruise_fraction = float(np.mean(cruise_mask))
    cruise_speed = float(np.mean(speeds[cruise_mask])) if np.any(cruise_mask) else mean_speed

    # Speed zero fraction (stopped/idle)
    zero_fraction = float(np.mean(speeds == 0))

    # Speed variability in first vs second half
    half = n // 2
    if half > 0:
        first_half_mean = float(np.mean(speeds[:half]))
        second_half_mean = float(np.mean(speeds[half:]))
    else:
        first_half_mean = mean_speed
        second_half_mean = mean_speed

    # Number of speed readings
    n_speed_points = n

    return {
        # Trip-level speed indicators
        "n_speed_points": n_speed_points,
        "mean_speed": round(mean_speed, 2),
        "max_speed_from_profile": round(max_speed, 2),
        "min_speed": round(min_speed, 2),
        "p85_speed": round(p85_speed, 2),
        "p95_speed": round(p95_speed, 2),
        "speed_std": round(speed_std, 2),
        "speed_cv": round(speed_cv, 4),

        # Speeding indicators
        "speeding_rate_25": round(speeding_rate_25, 4),
        "speeding_count_25": speeding_count_25,
        "speeding_duration_25_s": round(speeding_duration_25, 1),
        "speeding_rate_15": round(speeding_rate_15, 4),
        "speeding_rate_20": round(speeding_rate_20, 4),
        "speeding_rate_30": round(speeding_rate_30, 4),
        "max_excess_25": round(max_excess_25, 2),

        # Acceleration/deceleration
        "mean_abs_accel_ms2": round(mean_accel, 4),
        "max_accel_ms2": round(max_accel, 4),
        "max_decel_ms2": round(max_decel, 4),
        "harsh_accel_count": harsh_accel_count,
        "harsh_decel_count": harsh_decel_count,
        "harsh_event_count": harsh_event_count,
        "harsh_event_rate": round(harsh_event_rate, 4),

        # Speed profile features
        "ramp_up_duration_s": round(ramp_up_duration, 1),
        "cruise_fraction": round(cruise_fraction, 4),
        "cruise_speed": round(cruise_speed, 2),
        "zero_speed_fraction": round(zero_fraction, 4),
        "first_half_mean_speed": round(first_half_mean, 2),
        "second_half_mean_speed": round(second_half_mean, 2),
    }


def _empty_indicators() -> dict:
    """Return empty indicators for trips with no valid speed data."""
    return {
        "n_speed_points": 0,
        "mean_speed": None, "max_speed_from_profile": None, "min_speed": None,
        "p85_speed": None, "p95_speed": None, "speed_std": None, "speed_cv": None,
        "speeding_rate_25": None, "speeding_count_25": None,
        "speeding_duration_25_s": None,
        "speeding_rate_15": None, "speeding_rate_20": None, "speeding_rate_30": None,
        "max_excess_25": None,
        "mean_abs_accel_ms2": None, "max_accel_ms2": None, "max_decel_ms2": None,
        "harsh_accel_count": None, "harsh_decel_count": None,
        "harsh_event_count": None, "harsh_event_rate": None,
        "ramp_up_duration_s": None, "cruise_fraction": None,
        "cruise_speed": None, "zero_speed_fraction": None,
        "first_half_mean_speed": None, "second_half_mean_speed": None,
    }


def process_all_trips() -> None:
    """Process all trips in chunks and save indicators."""
    import duckdb

    con = duckdb.connect()
    parquet_path = str(DATA_DIR / "cleaned" / "trips_cleaned.parquet")

    print("=" * 70)
    print("  TASKS 2.1-2.3: COMPUTE TRIP-LEVEL INDICATORS")
    print("=" * 70)

    total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]
    print(f"\nTotal trips to process: {total:,}")

    # Process in chunks
    output_path = DATA_DIR / "trip_indicators.parquet"
    writer = None
    processed = 0
    parse_errors = 0

    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * CHUNK_SIZE

        chunk_df = con.execute(f"""
            SELECT route_id, speeds_raw
            FROM read_parquet('{parquet_path}')
            LIMIT {CHUNK_SIZE} OFFSET {offset}
        """).fetchdf()

        if len(chunk_df) == 0:
            break

        # Compute indicators for each trip
        indicators_list = []
        for _, row in chunk_df.iterrows():
            speeds = parse_speeds_fast(row["speeds_raw"])
            if speeds is not None and len(speeds) > 0:
                ind = compute_trip_indicators(speeds)
            else:
                ind = _empty_indicators()
                parse_errors += 1
            ind["route_id"] = row["route_id"]
            indicators_list.append(ind)

        # Convert to DataFrame and write as Parquet
        ind_df = pd.DataFrame(indicators_list)
        table = pa.Table.from_pandas(ind_df)

        if writer is None:
            writer = pq.ParquetWriter(
                str(output_path), table.schema,
                compression="zstd"
            )

        writer.write_table(table)
        processed += len(chunk_df)
        print(f"  Processed {processed:,}/{total:,} trips "
              f"({processed/total:.0%}) [chunk {chunk_idx+1}/{n_chunks}]")

    if writer is not None:
        writer.close()

    print(f"\nDone. Parse errors: {parse_errors}")
    print(f"Output: {output_path}")

    # Verify and summarize
    print("\n--- Indicator Summary Statistics ---")
    summary = con.execute(f"""
        SELECT
            COUNT(*) as total,
            AVG(mean_speed) as avg_mean_speed,
            AVG(max_speed_from_profile) as avg_max_speed,
            AVG(p85_speed) as avg_p85_speed,
            AVG(speed_cv) as avg_speed_cv,
            AVG(speeding_rate_25) as avg_speeding_rate_25,
            AVG(speeding_duration_25_s) as avg_speeding_dur_25,
            AVG(mean_abs_accel_ms2) as avg_abs_accel,
            AVG(harsh_event_rate) as avg_harsh_rate,
            SUM(harsh_event_count) as total_harsh_events,
            AVG(cruise_fraction) as avg_cruise_frac,
            AVG(ramp_up_duration_s) as avg_ramp_up,
            AVG(zero_speed_fraction) as avg_zero_frac
        FROM read_parquet('{output_path}')
    """).fetchone()

    print(f"  Total trips:             {summary[0]:>12,}")
    print(f"  Mean avg speed:          {summary[1]:>12.2f} km/h")
    print(f"  Mean max speed:          {summary[2]:>12.2f} km/h")
    print(f"  Mean P85 speed:          {summary[3]:>12.2f} km/h")
    print(f"  Mean speed CV:           {summary[4]:>12.4f}")
    print(f"  Mean speeding rate (25): {summary[5]:>12.4f}")
    print(f"  Mean speeding dur (25):  {summary[6]:>12.1f} s")
    print(f"  Mean |accel| (m/s^2):    {summary[7]:>12.4f}")
    print(f"  Mean harsh event rate:   {summary[8]:>12.4f}")
    print(f"  Total harsh events:      {summary[9]:>12,.0f}")
    print(f"  Mean cruise fraction:    {summary[10]:>12.4f}")
    print(f"  Mean ramp-up duration:   {summary[11]:>12.1f} s")
    print(f"  Mean zero-speed frac:    {summary[12]:>12.4f}")

    # Speeding rate distribution
    print("\n--- Speeding Rate Distribution ---")
    spd_dist = con.execute(f"""
        SELECT
            CASE
                WHEN speeding_rate_25 = 0 THEN '0% (no speeding)'
                WHEN speeding_rate_25 <= 0.1 THEN '0-10%'
                WHEN speeding_rate_25 <= 0.25 THEN '10-25%'
                WHEN speeding_rate_25 <= 0.5 THEN '25-50%'
                WHEN speeding_rate_25 <= 0.75 THEN '50-75%'
                ELSE '75-100%'
            END as speeding_bin,
            COUNT(*) as cnt,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM read_parquet('{output_path}')) as pct
        FROM read_parquet('{output_path}')
        GROUP BY speeding_bin
        ORDER BY speeding_bin
    """).fetchdf()
    for _, row in spd_dist.iterrows():
        print(f"  {row['speeding_bin']:<20}: {row['cnt']:>10,} ({row['pct']:>5.1f}%)")

    con.close()

    print("\n" + "=" * 70)
    print("  TRIP INDICATORS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    process_all_trips()
