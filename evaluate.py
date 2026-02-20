#!/usr/bin/env python3
"""
evaluate.py - Benchmark trajectory generation across all spatial × temporal strategy combinations.

Generates 1,000 trajectories with 100 points each for every combination (5 spatial ×
8 temporal = 40 cells), times each combination over several runs, saves detailed logs,
and produces a LaTeX table in the assets/ folder.

Usage:
    python evaluate.py             # run from repo root
    uv run evaluate.py             # via uv

Physics Based Generation is a *combined* strategy that handles spatial and temporal
internally; it cannot accept an external temporal strategy.  Those cells are marked
with an em-dash (—) in the table and logged as "N/A (combined strategy)".

The OSM graph download is performed once and the strategy object is reused across
all temporal strategy columns to avoid redundant network requests.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import statistics
import sys
from datetime import datetime
from timeit import default_timer as timer
from typing import Callable
from trajgen.config import Config  # noqa: E402
from trajgen.point_generator import RandomPointGenerator2D  # noqa: E402
from trajgen.trajectory import Trajectory  # noqa: E402
from trajgen.trajectory_generator import TrajectoryGenerator  # noqa: E402
from trajgen.spatial_strategy import (  # noqa: E402
    EqualDistributionStrategy,
    FreespaceStrategy,
    OsmSamplingStrategy,
    RandomWalkStrategy,
)
from trajgen.combined_strategy import PhysicsInformedCombinedStrategy  # noqa: E402
from trajgen.temporal_strategy import (  # noqa: E402
    AccelerationTemporalStrategy,
    ConstantTemporalStrategy,
    VelocityTemporalStrategy,
    VariableTimeStepsTemporalStrategy,
)
from trajgen.temporal_strategy.timesteps import (  # noqa: E402
    ConstantTimeStepsTemporalStrategy,
)

# --------------------------------------------------------------------------- #
# Global benchmark parameters
# --------------------------------------------------------------------------- #
NUM_TRAJECTORIES: int = 1_000
NUM_POINTS: int = 100
NUM_TIMING_RUNS: int = 25
SEED: int = 42
_ROOT = "./"
# Bounding boxes: standard unit square and Munich OSM area
BBOX_DEFAULT: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
BBOX_OSM: tuple[float, float, float, float] = (
    11.655981,
    11.668341,
    48.035282,
    48.051693,
)

LOGS_DIR: str = os.path.join(_ROOT, "logs")
ASSETS_DIR: str = os.path.join(_ROOT, "assets")

# --------------------------------------------------------------------------- #
# Temporal strategy factories
# Each entry: (display_name, factory(config) -> temporal_strategy_or_None)
# The factory receives a freshly-built Config and may mutate it before
# constructing the strategy object.
# --------------------------------------------------------------------------- #


def _temporal_none(c: Config):
    return None


def _temporal_constant_time(c: Config):
    return ConstantTemporalStrategy(c)


def _temporal_constant_velocity(c: Config):
    c.velocity_mode = "fixed for dataset"
    c.velocity = 1.0
    return VelocityTemporalStrategy(c)


def _temporal_velocity_from_dist(c: Config):
    c.velocity_mode = "fixed for trajectory"
    c.velocity_distribution = "uniform"
    c.velocity_min = 0.5
    c.velocity_max = 2.0
    return VelocityTemporalStrategy(c)


def _temporal_constant_acceleration(c: Config):
    c.acceleration_mode = "fixed for dataset"
    c.acceleration = 0.1
    return AccelerationTemporalStrategy(c)


def _temporal_acceleration_from_dist(c: Config):
    c.acceleration_mode = "fixed for trajectory"
    c.acceleration_distribution = "uniform"
    c.acceleration_min = 0.0
    c.acceleration_max = 1.0
    return AccelerationTemporalStrategy(c)


def _temporal_constant_ts(c: Config):
    return ConstantTimeStepsTemporalStrategy(c)


def _temporal_ts_from_dist(c: Config):
    c.time_step = 1.0
    return VariableTimeStepsTemporalStrategy(c)


# Ordered list that defines both row/column order and display names
TEMPORAL_STRATEGIES: list[tuple[str, Callable]] = [
    ("None", _temporal_none),
    ("Constant Time", _temporal_constant_time),
    ("Constant Velocity", _temporal_constant_velocity),
    ("Velocity from Distribution", _temporal_velocity_from_dist),
    ("Constant Acceleration", _temporal_constant_acceleration),
    ("Acceleration from Distribution", _temporal_acceleration_from_dist),
    ("Constant TS", _temporal_constant_ts),
    ("TS from Distribution", _temporal_ts_from_dist),
]

# Ordered list of spatial strategies (matches table rows)
SPATIAL_STRATEGIES: list[str] = [
    "Random Walk",
    "Equal Point Distribution",
    "Physics Based Generation",
    "Freespace Trajectories",
    "OSM Sampling",
]

# Human-readable short names for LaTeX rotated column headers
TEMPORAL_SHORT_NAMES: list[str] = [
    "None",
    "Constant Time",
    "Constant Velocity",
    "Velocity from Distribution",
    "Constant Acceleration",
    "Acceleration from Distribution",
    "Constant TS",
    "TS from distribution",
]

# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #


def make_config(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    seed: int = SEED,
) -> Config:
    """Return a fresh Config for the given bounding box."""
    return Config(
        seed=seed,
        length_mode="fixed for dataset",
        length=NUM_POINTS,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        velocity_mode="fixed for dataset",
        velocity=1.0,
        acceleration_mode="fixed for dataset",
        acceleration=0.1,
        start_time_mode="fixed for dataset",
        start_time=0.0,
        time_step=1.0,
    )


def attach_point_generator(config: Config) -> None:
    """Attach a standard 2-D random point generator to *config* in-place."""
    config.point_generator = RandomPointGenerator2D(
        (config.x_min, config.x_max),
        (config.y_min, config.y_max),
        config.seed,
    )


# --------------------------------------------------------------------------- #
# Generation helpers
# --------------------------------------------------------------------------- #


def generate_batch(generator: TrajectoryGenerator, config: Config) -> list[Trajectory]:
    """Generate NUM_TRAJECTORIES trajectories with per-trajectory seeding."""
    base_seed = config.seed
    trajectories: list[Trajectory] = []
    for i in range(NUM_TRAJECTORIES):
        traj_seed = base_seed + i
        config.rng = random.Random(traj_seed)
        # Reseed inner point generator if present
        pg = config.point_generator
        while pg is not None:
            if hasattr(pg, "rng"):
                pg.rng = random.Random(traj_seed)
                break
            pg = getattr(pg, "inner", None)
        trajectories.append(generator.generate_trajectory(i))
    return trajectories


def time_combination(
    make_generator: Callable[[], tuple[TrajectoryGenerator, Config]],
    num_runs: int = NUM_TIMING_RUNS,
) -> tuple[list[float], list[Trajectory]]:
    """
    Run *num_runs* timed generations.

    Each run calls *make_generator* to build a fresh (generator, config) pair
    before starting the clock — matching the app's behaviour where
    ``construct_trajectory_generator`` is called inside every timed iteration.

    Returns (times_list, trajectories_from_last_run).
    """
    times: list[float] = []
    trajectories: list[Trajectory] = []
    for _ in range(num_runs):
        t0 = timer()
        generator, config = make_generator()
        trajectories = generate_batch(generator, config)
        times.append(timer() - t0)
    return times, trajectories


# --------------------------------------------------------------------------- #
# Trajectory serialisation
# --------------------------------------------------------------------------- #


def trajectories_to_csv_str(trajectories: list[Trajectory]) -> str:
    """Serialise a list of trajectories to a CSV string (traj_id, point_idx, x, y[, time])."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    sample_coords = list(trajectories[0].ls.coords)
    has_z = len(sample_coords[0]) >= 3
    has_time = trajectories[0].t is not None

    header = ["traj_id", "point_idx", "x", "y"]
    if has_z:
        header.append("z")
    if has_time:
        header.append("time")
    writer.writerow(header)

    for traj in trajectories:
        coords = list(traj.ls.coords)
        for idx, coord in enumerate(coords):
            row = [traj.id, idx, coord[0], coord[1]]
            if has_z:
                row.append(coord[2] if len(coord) >= 3 else "")
            if has_time and traj.t is not None:
                row.append(traj.t[idx] if idx < len(traj.t) else "")
            writer.writerow(row)

    return buf.getvalue()


# --------------------------------------------------------------------------- #
# LaTeX table
# --------------------------------------------------------------------------- #


def _fmt(entry: tuple[float, float] | None) -> str:
    """Format an (avg, std) timing pair for a table cell."""
    if entry is None:
        return r"---"
    avg_s, std_s = entry
    if avg_s < 1.0:
        return rf"{avg_s * 1_000:.1f}{{\scriptsize$\pm${std_s * 1_000:.1f}}}\,ms"
    return rf"{avg_s:.2f}{{\scriptsize$\pm${std_s:.2f}}}\,s"


def build_latex_table(results: dict[str, dict[str, tuple[float, float] | None]]) -> str:
    """Return the complete LaTeX table string."""
    lines: list[str] = []
    lines.append(r"\begin{table}[]")
    lines.append(r"    \centering")
    lines.append(
        rf"    \caption{{Time to create {NUM_TRAJECTORIES:,} trajectories with "
        rf"{NUM_POINTS} points each with different methods, all 2D}}"
    )
    lines.append(r"    \label{tab:creation_time}")
    lines.append(r"    \begin{tabularx}{\columnwidth}{X|cccccccc}")
    lines.append(r"        \toprule")
    lines.append(r"        &\multicolumn{8}{c}{\textbf{Temporal Strategy}}\\")

    # Column header with rotated labels
    header_cells = [r"\textbf{Spatial Strategy}"] + [
        rf"\rothead{{{name}}}" for name in TEMPORAL_SHORT_NAMES
    ]
    lines.append("        " + "&".join(header_cells) + r"\\")
    lines.append(r"        \midrule")

    for spatial in SPATIAL_STRATEGIES:
        row_cells = [spatial] + [
            _fmt(results.get(spatial, {}).get(t_name))
            for t_name, _ in TEMPORAL_STRATEGIES
        ]
        lines.append("        " + " & ".join(row_cells) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabularx}")
    lines.append(
        r"    \begin{minipage}{\columnwidth}\footnotesize"
        "\n"
        r"        $^\dagger$ Physics Based Generation is a combined"
        " spatial+temporal strategy; no external temporal strategy is applied"
        " (--- cells)."
        "\n"
        r"    \end{minipage}"
    )
    lines.append(r"\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #


def save_timing_line(
    path: str, label: str, run_idx: int, elapsed: float, ts: str
) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{ts} | {label} | run {run_idx:02d} | {elapsed:.6f} s\n")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def main() -> None:  # noqa: C901  (complexity is inherent in the benchmark loop)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    run_id = now_dt.strftime("%Y%m%d_%H%M%S")

    timing_log_path = os.path.join(LOGS_DIR, f"{run_id}_timing.txt")
    json_log_path = os.path.join(LOGS_DIR, f"{run_id}_config.json")
    latex_path = os.path.join(ASSETS_DIR, "creation_time_table.tex")

    # results[spatial_name][temporal_name] = avg_seconds | None
    results: dict[str, dict[str, float | None]] = {s: {} for s in SPATIAL_STRATEGIES}

    detailed_log: dict = {
        "run_id": run_id,
        "timestamp": now_str,
        "num_trajectories": NUM_TRAJECTORIES,
        "num_points": NUM_POINTS,
        "num_timing_runs": NUM_TIMING_RUNS,
        "bbox_default": BBOX_DEFAULT,
        "bbox_osm": BBOX_OSM,
        "combinations": {},
    }

    total = len(SPATIAL_STRATEGIES) * len(TEMPORAL_STRATEGIES)
    combo_idx = 0

    # For OSM: prime the osmnx disk cache with one graph download before the
    # timed loop so that every timed run just loads from cache (matching the
    # app where the graph is cached from the preview step).
    osm_cache_primed = False

    print(f"TrajGen evaluation  –  {now_str}")
    print(
        f"  {NUM_TRAJECTORIES:,} trajectories × {NUM_POINTS} points × "
        f"{NUM_TIMING_RUNS} timing runs"
    )
    print(
        f"  {len(SPATIAL_STRATEGIES)} spatial × {len(TEMPORAL_STRATEGIES)} temporal = {total} combinations\n"
    )

    for spatial_name in SPATIAL_STRATEGIES:
        is_osm = spatial_name == "OSM Sampling"
        is_physics = spatial_name == "Physics Based Generation"

        bbox = BBOX_OSM if is_osm else BBOX_DEFAULT

        # Prime OSM cache before the first OSM temporal column
        if is_osm and not osm_cache_primed:
            print(
                "  Priming OSM graph cache (one-time download) ...", end=" ", flush=True
            )
            try:
                _prime_cfg = make_config(*BBOX_OSM)
                _prime_cfg.osm_max_hops = 50
                _prime_cfg.osm_max_meters = 5000.0
                _prime_cfg.osm_jitter_std_m = 5.0
                OsmSamplingStrategy(_prime_cfg)
                osm_cache_primed = True
                print("done\n")
            except Exception as exc:
                print(f"WARNING – cache priming failed: {exc}")

        for temporal_name, temporal_factory in TEMPORAL_STRATEGIES:
            combo_idx += 1
            label = f"{spatial_name} x {temporal_name}"
            print(f"[{combo_idx:02d}/{total}] {label}", end=" ... ", flush=True)

            # ---------------------------------------------------------------- #
            # Physics Based: combined strategy, temporal cannot be attached
            # ---------------------------------------------------------------- #
            if is_physics and temporal_name != "None":
                results[spatial_name][temporal_name] = None
                detailed_log["combinations"][label] = {
                    "status": "N/A (combined strategy handles temporal internally)"
                }
                print("N/A")
                continue

            # ---------------------------------------------------------------- #
            # Build a factory that creates a fresh generator each timing run.
            # Matches the app: construct_trajectory_generator is called inside
            # every timed iteration of generate_trajectories_timed().
            # ---------------------------------------------------------------- #
            try:
                # Capture loop variables for the closure
                _bbox = bbox
                _temporal_factory = temporal_factory
                _spatial_name = spatial_name
                _is_osm = is_osm
                _is_physics = is_physics

                def make_generator(
                    bbox=_bbox,
                    tf=_temporal_factory,
                    sname=_spatial_name,
                    is_osm=_is_osm,
                    is_physics=_is_physics,
                ) -> tuple[TrajectoryGenerator, Config]:
                    cfg = make_config(*bbox)

                    if is_osm:
                        cfg.osm_max_hops = 50
                        cfg.osm_max_meters = 5000.0
                        cfg.osm_jitter_std_m = 5.0
                        spatial = OsmSamplingStrategy(cfg)
                        temporal = tf(cfg)
                        return (
                            TrajectoryGenerator(
                                cfg,
                                spatial_strategy=spatial,
                                temporal_strategy=temporal,
                            ),
                            cfg,
                        )

                    if is_physics:
                        combined = PhysicsInformedCombinedStrategy(cfg)
                        return (
                            TrajectoryGenerator(
                                cfg, spatial_strategy=None, combined_strategy=combined
                            ),
                            cfg,
                        )

                    # Standard spatial strategy
                    attach_point_generator(cfg)

                    if sname == "Random Walk":
                        spatial = RandomWalkStrategy(cfg)
                    elif sname == "Equal Point Distribution":
                        spatial = EqualDistributionStrategy(cfg)
                        spatial.reset_for_dataset(NUM_TRAJECTORIES)
                    elif sname == "Freespace Trajectories":
                        # FreespaceStrategy reads start/end from config; set
                        # them explicitly (defaults are None → AttributeError).
                        cfg.start_point_mode = "fixed for dataset"
                        cfg.start_point_x = cfg.x_min
                        cfg.start_point_y = cfg.y_min
                        cfg.end_point_mode = "fixed for dataset"
                        cfg.end_point_x = cfg.x_max
                        cfg.end_point_y = cfg.y_max
                        spatial = FreespaceStrategy(cfg)
                    else:
                        raise ValueError(f"Unknown spatial strategy: {sname!r}")

                    temporal = tf(cfg)
                    return (
                        TrajectoryGenerator(
                            cfg, spatial_strategy=spatial, temporal_strategy=temporal
                        ),
                        cfg,
                    )

                # ------------------------------------------------------------ #
                # Time the generation (strategy construction included each run)
                # ------------------------------------------------------------ #
                times, trajectories = time_combination(make_generator)

                avg_t = sum(times) / len(times)
                std_t = statistics.stdev(times) if len(times) > 1 else 0.0
                results[spatial_name][temporal_name] = (avg_t, std_t)

                print(
                    f"avg={avg_t:.4f}s  std={std_t:.4f}s  "
                    f"min={min(times):.4f}s  max={max(times):.4f}s"
                )

                # ------------------------------------------------------------ #
                # Persist timing log lines  (mirrors _6_generate.py format)
                # ------------------------------------------------------------ #
                for i, t in enumerate(times):
                    save_timing_line(timing_log_path, label, i + 1, t, now_str)

                # ------------------------------------------------------------ #
                # Persist trajectories (last run only) as CSV
                # ------------------------------------------------------------ #
                safe_label = (
                    label.replace(" ", "_")
                    .replace("/", "-")
                    .replace("(", "")
                    .replace(")", "")
                )
                traj_csv_path = os.path.join(
                    LOGS_DIR, f"{run_id}_{safe_label}_trajectories.csv"
                )
                with open(traj_csv_path, "w", encoding="utf-8") as fh:
                    fh.write(trajectories_to_csv_str(trajectories))

                # ------------------------------------------------------------ #
                # Record in detailed log
                # ------------------------------------------------------------ #
                detailed_log["combinations"][label] = {
                    "status": "ok",
                    "times_s": times,
                    "avg_s": avg_t,
                    "std_s": std_t,
                    "min_s": min(times),
                    "max_s": max(times),
                    "trajectories_csv": os.path.basename(traj_csv_path),
                }

            except Exception as exc:
                print(f"ERROR – {exc}")
                results[spatial_name][temporal_name] = None
                detailed_log["combinations"][label] = {"status": f"error: {exc}"}

    # ----------------------------------------------------------------------- #
    # Save JSON log  (mirrors _6_generate.py config JSON)
    # ----------------------------------------------------------------------- #
    with open(json_log_path, "w", encoding="utf-8") as fh:
        json.dump(detailed_log, fh, indent=2, ensure_ascii=True)
    print(f"\nJSON log  → {json_log_path}")

    # ----------------------------------------------------------------------- #
    # Save LaTeX table
    # ----------------------------------------------------------------------- #
    latex_src = build_latex_table(results)
    with open(latex_path, "w", encoding="utf-8") as fh:
        fh.write(latex_src)
    print(f"LaTeX table → {latex_path}")

    # ----------------------------------------------------------------------- #
    # Console summary
    # ----------------------------------------------------------------------- #
    col_w = 14
    print("\n" + "=" * (30 + col_w * len(TEMPORAL_STRATEGIES)))
    print("RESULTS SUMMARY")
    print("=" * (30 + col_w * len(TEMPORAL_STRATEGIES)))
    header_label = "Spatial \\ Temporal"
    print(f"{header_label:<30}", end="")
    for t_name, _ in TEMPORAL_STRATEGIES:
        print(f"{t_name[:col_w-1]:<{col_w}}", end="")
    print()
    print("-" * (30 + col_w * len(TEMPORAL_STRATEGIES)))
    for s_name in SPATIAL_STRATEGIES:
        print(f"{s_name:<30}", end="")
        for t_name, _ in TEMPORAL_STRATEGIES:
            val = results[s_name].get(t_name)
            if val is None:
                cell = "N/A"
            else:
                avg, std = val
                if avg < 1.0:
                    cell = f"{avg * 1000:.1f}±{std * 1000:.1f}ms"
                else:
                    cell = f"{avg:.2f}±{std:.2f}s"
            print(f"{cell:<{col_w}}", end="")
        print()

    print(f"\nRun ID: {run_id}")


if __name__ == "__main__":
    main()
