"""
Task 3.2: Evaluate map-matching libraries on a sample of trips.

Compares LeuvenMapMatching and mappymatch (NREL) on 1,000 Seoul trips.
Evaluates: match rate, speed, matched distance vs reported distance.

Outputs:
  - data_parquet/modeling/map_matching_evaluation.json — comparison results
  - figures/map_matching_evaluation.png — summary comparison chart
"""

import ast
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    CLEANED_PARQUET,
    DATA_DIR,
    FIGURES_DIR,
    MODELING_DIR,
    OSM_NETWORKS_DIR,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Helper: parse trajectory from routes_raw string
# ---------------------------------------------------------------------------

def parse_trajectory(routes_raw: str) -> list[tuple[float, float]]:
    """Parse routes_raw string into list of (lat, lon) tuples.

    Args:
        routes_raw: Stringified list like
            [['2023/05/01','00:00:25.545', 37.497, 126.952], ...]

    Returns:
        List of (lat, lon) tuples.
    """
    try:
        points = ast.literal_eval(routes_raw)
        coords = [(float(p[2]), float(p[3])) for p in points]
        return coords
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sample trips
# ---------------------------------------------------------------------------

def sample_trips(
    city: str = "Seoul",
    n: int = 1000,
    min_points: int = 5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sample n valid trips from a city.

    Args:
        city: City to sample from.
        n: Number of trips to sample.
        min_points: Minimum GPS points per trip.
        seed: Random seed.

    Returns:
        DataFrame with route_id, routes_raw, distance, gps_points.
    """
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT route_id, routes_raw, distance, gps_points, moved_distance
        FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY RANDOM()) as rn
            FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
            WHERE is_valid = true
              AND city = '{city}'
              AND gps_points >= {min_points}
        )
        WHERE rn <= {n}
    """).fetchdf()
    con.close()
    print(f"Sampled {len(df)} trips from {city} (>= {min_points} GPS points)")
    return df


# ---------------------------------------------------------------------------
# Load OSM graph for a city
# ---------------------------------------------------------------------------

def load_osm_graph(city: str) -> Any:
    """Load OSM network from GeoPackage as a NetworkX graph.

    Args:
        city: City name (matches filename in osm_networks/).

    Returns:
        osmnx-compatible NetworkX MultiDiGraph.
    """
    gpkg_path = OSM_NETWORKS_DIR / f"{city}.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"No OSM network for {city}: {gpkg_path}")

    # Load from GeoPackage using geopandas
    nodes_gdf = gpd.read_file(gpkg_path, layer="nodes")
    edges_gdf = gpd.read_file(gpkg_path, layer="edges")

    # Restore osmid-based index for nodes
    if "osmid" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.set_index("osmid")
    nodes_gdf.index.name = "osmid"
    nodes_gdf.index = nodes_gdf.index.astype(np.int64)

    # Restore MultiIndex (u, v, key) for edges
    for col in ["u", "v", "key"]:
        if col in edges_gdf.columns:
            edges_gdf[col] = edges_gdf[col].astype(np.int64)
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    G = ox.graph_from_gdfs(nodes_gdf, edges_gdf)
    return G


# ---------------------------------------------------------------------------
# LeuvenMapMatching evaluation
# ---------------------------------------------------------------------------

def evaluate_leuven(
    trips_df: pd.DataFrame,
    G: Any,
    max_trips: int = 1000,
) -> dict:
    """Evaluate LeuvenMapMatching on a sample of trips.

    Args:
        trips_df: DataFrame with route_id, routes_raw.
        G: OSM NetworkX graph.
        max_trips: Maximum number of trips to process.

    Returns:
        Dict with results: match_rate, avg_time, distance_ratios, etc.
    """
    from leuvenmapmatching.map.inmem import InMemMap
    from leuvenmapmatching.matcher.distance import DistanceMatcher

    print("\n--- LeuvenMapMatching Evaluation ---")

    # Build Leuven InMemMap from OSM graph
    print("Building InMemMap from OSM graph...")
    t0 = time.time()
    leuven_map = InMemMap("osm", use_latlon=True, use_rtree=True, index_edges=True)

    # Add nodes
    for node_id, data in G.nodes(data=True):
        leuven_map.add_node(node_id, (data["y"], data["x"]))

    # Add edges
    for u, v, data in G.edges(data=True):
        leuven_map.add_edge(u, v)
        # Add reverse edge if not one-way
        if not data.get("oneway", False):
            leuven_map.add_edge(v, u)

    build_time = time.time() - t0
    print(f"Map built in {build_time:.1f}s "
          f"({leuven_map.size()} nodes)")

    # Match trips
    results = []
    matched_count = 0
    total_time = 0.0
    n_processed = 0
    errors = 0

    for idx, row in trips_df.head(max_trips).iterrows():
        coords = parse_trajectory(row["routes_raw"])
        if len(coords) < 3:
            continue

        n_processed += 1
        t0 = time.time()
        try:
            matcher = DistanceMatcher(
                leuven_map,
                max_dist=200,        # max distance from road in meters
                max_dist_init=300,   # initial search radius
                obs_noise=50,        # GPS noise in meters
                obs_noise_ne=75,     # noise for non-emitting states
                min_prob_norm=0.001,
            )
            states, _ = matcher.match(coords)
            elapsed = time.time() - t0
            total_time += elapsed

            if states and len(states) > 0:
                matched_count += 1
                # Compute matched path distance
                matched_nodes = states
                matched_dist = 0.0
                for i in range(len(matched_nodes) - 1):
                    try:
                        u_node = matched_nodes[i]
                        v_node = matched_nodes[i + 1]
                        if G.has_edge(u_node, v_node):
                            edge_data = G[u_node][v_node]
                            # MultiDiGraph: edge_data is dict of keys
                            first_key = list(edge_data.keys())[0]
                            matched_dist += edge_data[first_key].get("length", 0)
                    except Exception:
                        pass

                # Extract matched highway types
                highway_types = []
                for i in range(len(matched_nodes) - 1):
                    try:
                        u_node = matched_nodes[i]
                        v_node = matched_nodes[i + 1]
                        if G.has_edge(u_node, v_node):
                            edge_data = G[u_node][v_node]
                            first_key = list(edge_data.keys())[0]
                            hw = edge_data[first_key].get("highway", "unknown")
                            if isinstance(hw, list):
                                hw = hw[0]
                            highway_types.append(hw)
                    except Exception:
                        pass

                results.append({
                    "route_id": row["route_id"],
                    "gps_points": row["gps_points"],
                    "reported_distance": row["distance"],
                    "matched_distance": matched_dist,
                    "n_matched_nodes": len(states),
                    "match_time_s": elapsed,
                    "highway_types": highway_types,
                    "status": "matched",
                })
            else:
                results.append({
                    "route_id": row["route_id"],
                    "status": "no_match",
                    "match_time_s": elapsed,
                })

        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            errors += 1
            results.append({
                "route_id": row["route_id"],
                "status": "error",
                "error": str(e)[:100],
                "match_time_s": elapsed,
            })

        if n_processed % 100 == 0:
            print(f"  Processed {n_processed}/{max_trips} "
                  f"(matched: {matched_count}, errors: {errors})")

    match_rate = matched_count / max(n_processed, 1)
    avg_time = total_time / max(n_processed, 1)

    # Distance ratio for matched trips
    matched_results = [r for r in results if r["status"] == "matched"
                       and r.get("matched_distance", 0) > 0
                       and r.get("reported_distance", 0) > 0]
    dist_ratios = [r["matched_distance"] / r["reported_distance"]
                   for r in matched_results]

    print(f"\n  Leuven Results:")
    print(f"    Processed: {n_processed}")
    print(f"    Matched: {matched_count} ({match_rate:.1%})")
    print(f"    Errors: {errors}")
    print(f"    Avg time per trip: {avg_time*1000:.1f} ms")
    if dist_ratios:
        print(f"    Distance ratio (matched/reported): "
              f"median={np.median(dist_ratios):.2f}, "
              f"mean={np.mean(dist_ratios):.2f}")

    return {
        "library": "LeuvenMapMatching",
        "n_processed": n_processed,
        "n_matched": matched_count,
        "n_errors": errors,
        "match_rate": round(match_rate, 4),
        "avg_time_ms": round(avg_time * 1000, 1),
        "total_time_s": round(total_time, 1),
        "build_time_s": round(build_time, 1),
        "dist_ratio_median": round(np.median(dist_ratios), 3) if dist_ratios else None,
        "dist_ratio_mean": round(np.mean(dist_ratios), 3) if dist_ratios else None,
        "dist_ratio_std": round(np.std(dist_ratios), 3) if dist_ratios else None,
        "detailed_results": results,
    }


# ---------------------------------------------------------------------------
# mappymatch evaluation
# ---------------------------------------------------------------------------

def evaluate_mappymatch(
    trips_df: pd.DataFrame,
    city: str = "Seoul",
    max_trips: int = 1000,
) -> dict:
    """Evaluate mappymatch (NREL) on a sample of trips.

    Args:
        trips_df: DataFrame with route_id, routes_raw.
        city: City name for network download.
        max_trips: Maximum number of trips to process.

    Returns:
        Dict with results: match_rate, avg_time, distance_ratios, etc.
    """
    from mappymatch import package_root
    from mappymatch.constructs.geofence import Geofence
    from mappymatch.constructs.trace import Trace
    from mappymatch.maps.nx.nx_map import NxMap
    from mappymatch.matchers.lcss.lcss import LCSSMatcher

    print("\n--- mappymatch (NREL) Evaluation ---")

    # Build map from first trip's bounding box (expanded)
    # Sample a few trips to get the bounding box
    all_coords = []
    for _, row in trips_df.head(50).iterrows():
        coords = parse_trajectory(row["routes_raw"])
        all_coords.extend(coords)

    if not all_coords:
        print("  ERROR: Could not parse any trajectories")
        return {"library": "mappymatch", "error": "no parseable trajectories"}

    lats = [c[0] for c in all_coords]
    lons = [c[1] for c in all_coords]

    # Build geofence from bounding box
    print("Building NxMap from geofence...")
    t0 = time.time()
    try:
        geofence = Geofence.from_bbox(
            min_x=min(lons) - 0.01,
            max_x=max(lons) + 0.01,
            min_y=min(lats) - 0.01,
            max_y=max(lats) + 0.01,
        )
        nx_map = NxMap.from_geofence(geofence)
        build_time = time.time() - t0
        print(f"  Map built in {build_time:.1f}s")
    except Exception as e:
        print(f"  ERROR building map: {e}")
        return {"library": "mappymatch", "error": str(e)}

    # Initialize matcher
    matcher = LCSSMatcher(nx_map)

    # Match trips
    results = []
    matched_count = 0
    total_time = 0.0
    n_processed = 0
    errors = 0

    for idx, row in trips_df.head(max_trips).iterrows():
        coords = parse_trajectory(row["routes_raw"])
        if len(coords) < 3:
            continue

        n_processed += 1
        t0 = time.time()

        try:
            # Build trace from coordinates
            trace_df = pd.DataFrame(coords, columns=["latitude", "longitude"])
            trace = Trace.from_dataframe(trace_df)

            # Run matching
            match_result = matcher.match_trace(trace)
            elapsed = time.time() - t0
            total_time += elapsed

            # Check result
            matched_edges = [m for m in match_result.matches if m.road is not None]
            if len(matched_edges) > 0:
                matched_count += 1
                # Compute matched distance
                matched_dist = sum(
                    m.road.geom.length * 111320  # approximate degrees to meters
                    for m in matched_edges
                    if m.road is not None and m.road.geom is not None
                )

                # Extract road types
                highway_types = []
                for m in matched_edges:
                    if m.road is not None and hasattr(m.road, "metadata"):
                        hw = m.road.metadata.get("highway", "unknown")
                        if isinstance(hw, list):
                            hw = hw[0]
                        highway_types.append(hw)

                results.append({
                    "route_id": row["route_id"],
                    "gps_points": row["gps_points"],
                    "reported_distance": row["distance"],
                    "matched_distance": matched_dist,
                    "n_matched_edges": len(matched_edges),
                    "match_coverage": len(matched_edges) / len(coords),
                    "match_time_s": elapsed,
                    "highway_types": highway_types,
                    "status": "matched",
                })
            else:
                results.append({
                    "route_id": row["route_id"],
                    "status": "no_match",
                    "match_time_s": elapsed,
                })

        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            errors += 1
            results.append({
                "route_id": row["route_id"],
                "status": "error",
                "error": str(e)[:100],
                "match_time_s": elapsed,
            })

        if n_processed % 100 == 0:
            print(f"  Processed {n_processed}/{max_trips} "
                  f"(matched: {matched_count}, errors: {errors})")

    match_rate = matched_count / max(n_processed, 1)
    avg_time = total_time / max(n_processed, 1)

    # Distance ratio
    matched_results = [r for r in results if r["status"] == "matched"
                       and r.get("matched_distance", 0) > 0
                       and r.get("reported_distance", 0) > 0]
    dist_ratios = [r["matched_distance"] / r["reported_distance"]
                   for r in matched_results]

    print(f"\n  mappymatch Results:")
    print(f"    Processed: {n_processed}")
    print(f"    Matched: {matched_count} ({match_rate:.1%})")
    print(f"    Errors: {errors}")
    print(f"    Avg time per trip: {avg_time*1000:.1f} ms")
    if dist_ratios:
        print(f"    Distance ratio (matched/reported): "
              f"median={np.median(dist_ratios):.2f}, "
              f"mean={np.mean(dist_ratios):.2f}")

    return {
        "library": "mappymatch",
        "n_processed": n_processed,
        "n_matched": matched_count,
        "n_errors": errors,
        "match_rate": round(match_rate, 4),
        "avg_time_ms": round(avg_time * 1000, 1),
        "total_time_s": round(total_time, 1),
        "build_time_s": round(build_time, 1),
        "dist_ratio_median": round(np.median(dist_ratios), 3) if dist_ratios else None,
        "dist_ratio_mean": round(np.mean(dist_ratios), 3) if dist_ratios else None,
        "dist_ratio_std": round(np.std(dist_ratios), 3) if dist_ratios else None,
        "detailed_results": results,
    }


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(leuven_results: dict, mappy_results: dict) -> None:
    """Create comparison figure for the two map-matching libraries.

    Args:
        leuven_results: Results dict from evaluate_leuven.
        mappy_results: Results dict from evaluate_mappymatch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    libraries = ["LeuvenMapMatching", "mappymatch"]
    colors = ["#2196F3", "#FF9800"]

    # Panel 1: Match rate
    ax = axes[0]
    rates = [
        leuven_results.get("match_rate", 0),
        mappy_results.get("match_rate", 0),
    ]
    bars = ax.bar(libraries, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Match Rate")
    ax.set_title("(a) Match Rate")
    ax.set_ylim(0, 1.05)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=11)

    # Panel 2: Speed (ms per trip)
    ax = axes[1]
    times = [
        leuven_results.get("avg_time_ms", 0),
        mappy_results.get("avg_time_ms", 0),
    ]
    bars = ax.bar(libraries, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Avg Time per Trip (ms)")
    ax.set_title("(b) Processing Speed")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.0f} ms", ha="center", va="bottom", fontsize=11)

    # Panel 3: Distance ratio distribution
    ax = axes[2]
    for lib_results, label, color in [
        (leuven_results, "Leuven", colors[0]),
        (mappy_results, "mappymatch", colors[1]),
    ]:
        matched = [r for r in lib_results.get("detailed_results", [])
                   if r.get("status") == "matched"
                   and r.get("matched_distance", 0) > 0
                   and r.get("reported_distance", 0) > 0]
        if matched:
            ratios = [r["matched_distance"] / r["reported_distance"]
                      for r in matched]
            # Clip extreme values for visualization
            ratios = [min(max(r, 0), 5) for r in ratios]
            ax.hist(ratios, bins=50, alpha=0.6, label=label, color=color,
                    edgecolor="black", linewidth=0.3)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1, label="Ideal (1.0)")
    ax.set_xlabel("Matched / Reported Distance")
    ax.set_ylabel("Count")
    ax.set_title("(c) Distance Ratio Distribution")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "map_matching_evaluation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "map_matching_evaluation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {FIGURES_DIR / 'map_matching_evaluation.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run map-matching evaluation."""
    print("=" * 60)
    print("Task 3.2: Evaluate Map-Matching Libraries")
    print("=" * 60)

    EVAL_CITY = "Seoul"
    N_SAMPLE = 200
    N_LEUVEN = 200
    N_MAPPY = 100

    # Step 1: Sample trips
    print(f"\n1. Sampling {N_SAMPLE} trips from {EVAL_CITY}...")
    trips_df = sample_trips(city=EVAL_CITY, n=N_SAMPLE)

    # Step 2: Load OSM graph
    print(f"\n2. Loading OSM graph for {EVAL_CITY}...")
    G = load_osm_graph(EVAL_CITY)
    print(f"   Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Step 3: Evaluate LeuvenMapMatching
    print(f"\n3. Evaluating LeuvenMapMatching on {N_LEUVEN} trips...")
    leuven_results = evaluate_leuven(trips_df, G, max_trips=N_LEUVEN)

    # Step 4: Evaluate mappymatch
    print(f"\n4. Evaluating mappymatch on {N_MAPPY} trips...")
    mappy_results = evaluate_mappymatch(trips_df, city=EVAL_CITY, max_trips=N_MAPPY)

    # Step 5: Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Leuven':<20} {'mappymatch':<20}")
    print("-" * 70)
    print(f"{'Match rate':<30} {leuven_results.get('match_rate', 0):.1%}"
          f"{'':<14} {mappy_results.get('match_rate', 0):.1%}")
    print(f"{'Avg time (ms)':<30} {leuven_results.get('avg_time_ms', 0):.1f}"
          f"{'':<14} {mappy_results.get('avg_time_ms', 0):.1f}")
    print(f"{'Dist ratio median':<30} {leuven_results.get('dist_ratio_median', 'N/A')}"
          f"{'':<14} {mappy_results.get('dist_ratio_median', 'N/A')}")
    print(f"{'Errors':<30} {leuven_results.get('n_errors', 0)}"
          f"{'':<14} {mappy_results.get('n_errors', 0)}")

    # Step 6: Save results
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    # Remove detailed_results for JSON (too large)
    leuven_summary = {k: v for k, v in leuven_results.items()
                      if k != "detailed_results"}
    mappy_summary = {k: v for k, v in mappy_results.items()
                     if k != "detailed_results"}

    evaluation = {
        "eval_city": EVAL_CITY,
        "n_sample": N_SAMPLE,
        "leuven": leuven_summary,
        "mappymatch": mappy_summary,
        "recommendation": "",  # Will be filled after analysis
    }

    # Determine recommendation
    l_rate = leuven_results.get("match_rate", 0)
    m_rate = mappy_results.get("match_rate", 0)
    l_time = leuven_results.get("avg_time_ms", float("inf"))
    m_time = mappy_results.get("avg_time_ms", float("inf"))

    if l_rate > m_rate and l_time < m_time:
        evaluation["recommendation"] = "LeuvenMapMatching (better rate and speed)"
    elif m_rate > l_rate and m_time < l_time:
        evaluation["recommendation"] = "mappymatch (better rate and speed)"
    elif l_rate > m_rate:
        evaluation["recommendation"] = "LeuvenMapMatching (better match rate)"
    elif l_time < m_time:
        evaluation["recommendation"] = "LeuvenMapMatching (faster processing)"
    else:
        evaluation["recommendation"] = "mappymatch (overall better performance)"

    output_path = MODELING_DIR / "map_matching_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Step 7: Plot comparison
    print("\n5. Creating comparison figure...")
    plot_comparison(leuven_results, mappy_results)

    print(f"\nRecommendation: {evaluation['recommendation']}")


if __name__ == "__main__":
    main()
