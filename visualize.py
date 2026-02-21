#!/usr/bin/env python3
"""
visualize.py — one sample trajectory per 2D spatial sampling method saved as SVG.

Run from the repository root:
    python visualize.py

Output: assets/trajectory_methods_<run_id>.svg
        assets/trajectory_methods_<run_id>_config.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from shapely.geometry import Point as ShapelyPoint  # noqa: E402

from trajgen.combined_strategy import PhysicsInformedCombinedStrategy  # noqa: E402
from trajgen.config import Config  # noqa: E402
from trajgen.point_generator import RandomPointGenerator2D  # noqa: E402
from trajgen.spatial_strategy import (  # noqa: E402
    EqualDistributionStrategy,
    FreespaceStrategy,
    OsmSamplingStrategy,
    RandomWalkStrategy,
)
from trajgen.trajectory_generator import TrajectoryGenerator  # noqa: E402

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
SEED = 42
TRAJ_ID = 0
NUM_POINTS = 15
BBOX_DEFAULT = (0.0, 1.0, 0.0, 1.0)
BBOX_OSM = (11.655981, 11.668341, 48.035282, 48.051693)
PAGEWIDTH_INCHES = 7.165354  # A4 width with 0.5 inch margins on each side
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Physics panel: fixed launch position (centre of box)
PHYSICS_START = (0.5, 0.5)
PHYSICS_TRAJ_ID = 1

# --------------------------------------------------------------------------- #
# Colour palette
# --------------------------------------------------------------------------- #
# TUM Colours
C_TRAJ = "#005293"  # blue
C_START = "#A2AD00"  # green
C_END = "#E37222"  # red
C_OBS_FILL = "#DAD7CB"
C_OBS_EDGE = "#808080"
C_ROAD = "#DAD7CB"  # light grey
C_PT = "#DAD7CB"  # red

# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #


def _make_config(x_min: float, x_max: float, y_min: float, y_max: float) -> Config:
    cfg = Config({}, seed=SEED)
    cfg.x_min = x_min
    cfg.x_max = x_max
    cfg.y_min = y_min
    cfg.y_max = y_max
    cfg.get_next_length = NUM_POINTS
    cfg.get_next_length_mode = "fixed for dataset"
    cfg.get_next_closed_loop = False
    cfg.get_next_closed_loop_mode = "fixed for dataset"
    return cfg


def _attach_point_gen(cfg: Config) -> None:
    cfg.point_generator = RandomPointGenerator2D(
        (cfg.x_min, cfg.x_max), (cfg.y_min, cfg.y_max), cfg.seed
    )


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def _draw_trajectory(ax, traj, *, point_size: float = 40, draw_points: bool = True):
    coords = list(traj.ls.coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    ax.plot(xs, ys, "-", color=C_TRAJ, linewidth=1.6, zorder=3)

    if draw_points:
        ax.scatter(
            xs,
            ys,
            s=point_size,
            color=C_TRAJ,
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.scatter(xs[0], ys[0], s=75, color=C_START, zorder=5, marker="^")
    ax.scatter(xs[-1], ys[-1], s=75, color=C_END, zorder=5, marker="s")


def _finish_ax(ax, title: str, xlim=None, ylim=None, grid: bool = True):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.3, linewidth=0.4, zorder=0)


# --------------------------------------------------------------------------- #
# Per-method panels
# --------------------------------------------------------------------------- #


def _panel_random_walk(ax) -> None:
    cfg = _make_config(*BBOX_DEFAULT)
    _attach_point_gen(cfg)
    spatial = RandomWalkStrategy(cfg)
    traj = TrajectoryGenerator(cfg, spatial).generate_trajectory(TRAJ_ID)
    _draw_trajectory(ax, traj)
    _finish_ax(ax, "Constrained Random", (-0.05, 1.05), (-0.05, 1.05))
    return cfg


def _panel_equal_distribution(ax) -> None:
    cfg = _make_config(*BBOX_DEFAULT)
    _attach_point_gen(cfg)
    cfg.grid_rows = 5
    cfg.grid_cols = 5
    spatial = EqualDistributionStrategy(cfg)
    spatial.reset_for_dataset(1)
    traj = TrajectoryGenerator(cfg, spatial).generate_trajectory(TRAJ_ID)

    # Cell grid
    rows, cols = cfg.grid_rows, cfg.grid_cols
    for i in range(rows + 1):
        ax.axhline(i / rows, color="#e5e7eb", linewidth=0.6, zorder=1)
    for j in range(cols + 1):
        ax.axvline(j / cols, color="#e5e7eb", linewidth=0.6, zorder=1)

    _draw_trajectory(ax, traj)
    _finish_ax(ax, "Environment Weighted", (-0.02, 1.02), (-0.02, 1.02), grid=False)
    return cfg


def _panel_freespace(ax) -> None:
    cfg = _make_config(*BBOX_DEFAULT)
    _attach_point_gen(cfg)
    cfg.get_start_point_mode = "fixed for dataset"
    cfg.get_start_point_x = 0.00
    cfg.get_start_point_y = 0.00
    cfg.get_end_point_mode = "fixed for dataset"
    cfg.get_end_point_x = 1.0
    cfg.get_end_point_y = 1
    cfg.num_obstacles = 5
    cfg.obstacle_size_min = 0.06
    cfg.obstacle_size_max = 0.14
    cfg.deviation_factor = 0.15
    spatial = FreespaceStrategy(cfg)
    traj = TrajectoryGenerator(cfg, spatial).generate_trajectory(TRAJ_ID)

    for obs in spatial.obstacles:
        xo, yo = obs.polygon.exterior.xy
        ax.fill(xo, yo, color=C_OBS_FILL, alpha=0.6, zorder=2)
        ax.plot(xo, yo, color=C_OBS_EDGE, linewidth=0.7, zorder=2)

    _draw_trajectory(ax, traj)
    _finish_ax(ax, "Constrained Freespace", (-0.02, 1.02), (-0.02, 1.02))
    return cfg


def _panel_physics(ax) -> None:
    # Seed the global RNG used by PhysicsInformedCombinedStrategy for
    # reproducible initial conditions (it uses random.uniform, not config.rng).
    random.seed(SEED)
    cfg = _make_config(*BBOX_DEFAULT)
    # Fix the launch position:
    cfg.point_generator = lambda n: [ShapelyPoint(*PHYSICS_START)] * n
    cfg.gravity = 9.81
    cfg.simulation_time = 1.0
    cfg.bounce_damping = 0.7
    cfg.num_balls = 1
    # Initial velocity: applied to both vx and vy.
    cfg.get_next_velocity = 5.0
    cfg.time_step = 0.05
    cfg.get_next_velocity_mode = "fixed for dataset"
    combined = PhysicsInformedCombinedStrategy(cfg)
    traj = TrajectoryGenerator(
        cfg, None, combined_strategy=combined
    ).generate_trajectory(PHYSICS_TRAJ_ID)
    print(traj)
    _draw_trajectory(ax, traj)
    _finish_ax(ax, "Physics Informed", (-0.05, 1.05), (-0.05, 1.05))
    return cfg


def _panel_osm(ax) -> None:
    cfg = _make_config(*BBOX_OSM)
    cfg.osm_max_hops = 50
    cfg.osm_max_meters = 5000.0
    cfg.osm_max_attempts_per_traj = 10
    spatial = OsmSamplingStrategy(cfg)
    traj = TrajectoryGenerator(cfg, spatial).generate_trajectory(TRAJ_ID)

    # Road network edges — plot from unprojected WGS-84 graph (lon/lat == x/y)
    G = spatial._G
    for u, v, data in G.edges(data=True):
        if "geometry" in data:
            xs, ys = data["geometry"].xy
        else:
            xs = [G.nodes[u]["x"], G.nodes[v]["x"]]
            ys = [G.nodes[u]["y"], G.nodes[v]["y"]]
        ax.plot(xs, ys, color=C_ROAD, linewidth=0.5, zorder=1)

    # PT hotspot nodes
    for node_id in spatial._hotspots:
        nd = G.nodes[node_id]
        ax.scatter(
            nd["x"],
            nd["y"],
            s=14,
            color=C_PT,
            zorder=3,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
        )

    _draw_trajectory(ax, traj, point_size=18)
    _finish_ax(ax, "Map Based", grid=False)
    ax.set_xlabel("lon", fontsize=6, labelpad=2)
    ax.set_ylabel("lat", fontsize=6, labelpad=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    return cfg


# --------------------------------------------------------------------------- #
# Legend
# --------------------------------------------------------------------------- #


def _legend_handles():
    return [
        Line2D([0], [0], color=C_TRAJ, linewidth=1.6, label="Trajectory"),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            color="w",
            markerfacecolor=C_START,
            markersize=7,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="w",
            markerfacecolor=C_END,
            markersize=7,
            label="End",
        ),
        mpatches.Patch(
            facecolor=C_OBS_FILL,
            edgecolor=C_OBS_EDGE,
            label="Obstacle (Freespace)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="w",
            markerfacecolor=C_PT,
            markersize=6,
            label="PT hotspot (OSM)",
        ),
        Line2D([0], [0], color=C_ROAD, linewidth=1.5, label="Road network (OSM)"),
    ]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

PANELS = [
    ("Constrained Random", _panel_random_walk),
    ("Environment Weighted", _panel_equal_distribution),
    ("Physics Informed", _panel_physics),
    ("Constrained Freespace", _panel_freespace),
    ("Map Based", _panel_osm),
]


def main() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 9
    os.makedirs(ASSETS_DIR, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(ASSETS_DIR, f"trajectory_methods_{run_id}.pdf")
    out_config_path = os.path.join(
        ASSETS_DIR, f"trajectory_methods_{run_id}_config_"
    )

    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(PAGEWIDTH_INCHES, 0.3 * PAGEWIDTH_INCHES),
        gridspec_kw={"wspace": 0.40},
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.3)
    # fig.suptitle(
    #     "Spatial Trajectory Sampling Methods",
    #     fontsize=11,
    #     fontweight="bold",
    #     y=0.97,
    # )

    for i, (name, fn) in enumerate(PANELS):
        print(f"  [{i + 1}/{len(PANELS)}] {name} ...", flush=True)
        cfg = fn(axes[i])
        cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
        cfg_dict["seed"] = cfg.seed  # include seed explicitly
        cfg_dict["spatial_strategy"] = name
        with open(out_config_path+name+".json", "w", encoding="utf-8") as fh:
            json.dump(cfg_dict, fh, indent=2)

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=7.5,
        frameon=True,
        framealpha=0.9,
        handlelength=1.6,
    )

    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.close(fig)

    # ----------------------------------------------------------------------- #
    # Config file — records every parameter used to produce the SVG
    # ----------------------------------------------------------------------- #


    print(f"Config  → {out_config_path}")


if __name__ == "__main__":
    main()
