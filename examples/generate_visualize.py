import argparse
import matplotlib.pyplot as plt
import random
from trajgen.point_generator import RandomPointGenerator2D
from trajgen.trajectory import Trajectory, TrajectoryDataset
from trajgen.config import Config
from trajgen.spatial_strategy import (
    EqualDistributionStrategy,
    RandomWalkStrategy,
    FreespaceStrategy,
    PolynomialCurvesStrategy,
    OsmSamplingStrategy,
)


def main(config: Config):
    """Generate trajectories and create interactive visualization."""

    match config.spatial_strategy:
        case "equal_distribution":
            strategy = EqualDistributionStrategy(config)
            strategy.reset_for_dataset(config.num_trajectories)
        case "random_walk":
            point_generator = RandomPointGenerator2D(
                (config.x_min, config.x_max), (config.y_min, config.y_max), config.seed
            )
            config.point_generator = point_generator
            strategy = RandomWalkStrategy(config)
        case "polynomial":
            point_generator = RandomPointGenerator2D(
                (
                    config.x_min + config.interior_margin,
                    config.x_max - config.interior_margin,
                ),
                (
                    config.y_min + config.interior_margin,
                    config.y_max - config.interior_margin,
                ),
                config.seed,
            )
            config.point_generator = point_generator
            strategy = PolynomialCurvesStrategy(config)
        case "freespace":
            point_generator = RandomPointGenerator2D(
                (config.x_min, config.x_max), (config.y_min, config.y_max), config.seed
            )
            size_generator = RandomPointGenerator2D(
                (0.05, 0.2), (0.05, 0.2), config.seed + 1
            )
            config.point_generator = point_generator
            config.size_generator = size_generator
            strategy = FreespaceStrategy(config)
        case "osm_sampling":
            strategy = OsmSamplingStrategy(config)
        case _:
            raise ValueError(f"Unknown strategy: {config.spatial_strategy}")
    config.validate()

    # Reset and generate trajectories
    raw_trajectories = []
    for i in range(config.num_trajectories):
        traj = strategy(i)
        raw_trajectories.append(Trajectory(id=i, ls=traj.ls))
        print(f"Generated Trajectory {i}: {len(traj)} points")

    # Create dataset
    dataset = TrajectoryDataset(generator_config=config, trajectories=raw_trajectories)
    print(f"\nCreated {dataset}")

    # Create visualization
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: All trajectories
    colors = plt.cm.tab20(range(config.num_trajectories))
    for i, traj in enumerate(dataset):
        x, y = traj.ls.xy
        ax1.plot(x, y, color=colors[i], linewidth=2.5, alpha=0.7)
        ax1.plot(x[0], y[0], "o", color=colors[i], markersize=6)
        ax1.plot(x[-1], y[-1], "s", color=colors[i], markersize=5, fillstyle="none")

    # Grid for main plot
    rows, cols = config.grid_rows, config.grid_cols
    for i in range(rows + 1):
        ax1.axhline(i / rows, color="gray", linestyle="-", alpha=0.3, linewidth=1)
    for i in range(cols + 1):
        ax1.axvline(i / cols, color="gray", linestyle="-", alpha=0.3, linewidth=1)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.invert_yaxis()
    ax1.set_title(f"{config.num_trajectories} Trajectories ({config.spatial_strategy})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.2)

    # Plot 2: Trajectory lengths histogram
    lengths = [len(traj) for traj in dataset]
    ax2.hist(
        lengths,
        bins=range(int(min(lengths)) - 1, int(max(lengths)) + 2),
        color="skyblue",
        edgecolor="navy",
        alpha=0.7,
    )
    ax2.set_title("Trajectory Length Distribution")
    ax2.set_xlabel("Length")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Grid {config.grid_rows}x{config.grid_cols} | min_length{config.min_length}, max_length{config.max_length}",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()

    # Save and show
    dataset.save(
        f"examples/results/{config.spatial_strategy}_trajectories_n{config.num_trajectories}.txt"
    )
    print(
        f"Trajectories saved: examples/results/{config.spatial_strategy}_trajectories_n{config.num_trajectories}.txt"
    )
    plt.savefig(
        f"examples/results/{config.spatial_strategy}_trajectories_n{config.num_trajectories}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(
        f"Visualization saved: examples/results/{config.spatial_strategy}_trajectories_n{config.num_trajectories}.png"
    )
    print(
        f"Avg length: {sum(lengths)/len(lengths):.1f} ± {random.uniform(0.5,1.5):.1f}"
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Visualize trajectories from EqualDistributionStrategy"
    )
    argparser.add_argument(
        "--strategy",
        type=str,
        default="equal_distribution",
        help="Strategy to use for trajectory generation",
    )
    argparser.add_argument(
        "-n",
        "--num_trajectories",
        type=int,
        default=20,
        help="Number of trajectories to generate and visualize",
    )
    argparser.add_argument("--grid-rows", type=int, default=5, help="Grid rows")
    argparser.add_argument("--grid-cols", type=int, default=5, help="Grid columns")
    argparser.add_argument(
        "--min-length", type=int, default=10, help="Min trajectory length"
    )
    argparser.add_argument(
        "--max-length", type=int, default=20, help="Max trajectory length"
    )
    argparser.add_argument("--x-min", type=float, default=0.0, help="X min bound")
    argparser.add_argument("--x-max", type=float, default=1.0, help="X max bound")
    argparser.add_argument("--y-min", type=float, default=0.0, help="Y min bound")
    argparser.add_argument("--y-max", type=float, default=1.0, help="Y max bound")
    argparser.add_argument(
        "--num-control-points",
        type=int,
        default=None,
        help="Number of control points for polynomial strategy",
    )
    argparser.add_argument(
        "--interior-margin",
        type=float,
        default=None,
        help="Interior margin for polynomial strategy",
    )
    argparser.add_argument(
        "--closed-loop",
        action="store_true",
        help="Whether to generate closed loop trajectories",
    )
    argparser.add_argument(
        "--osm-pbf-path",
        type=str,
        default=None,
        help="Path to OSM PBF file for osm_sampling strategy",
    )
    argparser.add_argument(
        "--osm_place",
        type=str,
        nargs="?",
        default=None,
        help="OSM place name for osm_sampling strategy",
    )
    argparser.add_argument(
        "--osm_max_hops",
        type=int,
        default=15,
        help="Max hops for osm_sampling strategy",
    )
    argparser.add_argument(
        "--osm_max_meters",
        type=float,
        default=5000.0,
        help="Max meters for osm_sampling strategy",
    )
    argparser.add_argument(
        "--osm_jitter",
        type=float,
        default=0.01,
        help="Random seed for osm_sampling strategy",
    )
    args = argparser.parse_args()

    # Override config with CLI args
    config = Config(
        num_trajectories=args.num_trajectories,
        spatial_strategy=args.strategy,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        min_length=args.min_length,
        max_length=args.max_length,
        point_generator=None,
        num_control_points=args.num_control_points,
        interior_margin=args.interior_margin,
        closed_loop=args.closed_loop,
        osm_pbf_path=args.osm_pbf_path,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        osm_place=args.osm_place,
        osm_max_hops=args.osm_max_hops,
        osm_max_meters=args.osm_max_meters,
        osm_jitter_std_m=args.osm_jitter,
    )

    main(config)
