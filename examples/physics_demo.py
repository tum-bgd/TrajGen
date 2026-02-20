#!/usr/bin/env python3
"""
Demo script for the Physics-Informed Combined Strategy.

This script demonstrates the physics simulation of bouncing balls
in a bounding box with gravity and generates visualization images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from shapely.geometry import LineString
from trajgen.combined_strategy.physics_informed import (
    PhysicsInformedCombinedStrategy,
)
from trajgen.trajectory import Trajectory


class DemoConfig:
    """Demo configuration for physics simulation."""

    def __init__(self, dimension="2D"):
        # Dimension configuration
        self.dimension = dimension

        # Bounding box (meters)
        self.x_min = 0.0
        self.x_max = 50.0
        self.y_min = 0.0  # floor
        self.y_max = 25.0  # ceiling

        # 3D-specific bounds
        if dimension == "3D":
            self.z_min = 0.0
            self.z_max = 30.0

        # Physics parameters
        self.gravity = 9.81  # m/s^2
        self.time_step = 0.02  # 20ms time steps
        self.simulation_time = 10.0  # 10 seconds
        self.bounce_damping = 0.85  # Energy retained after bounce
        self.num_balls = 1

        # Visualization parameters
        self.output_dir = Path(
            f"examples/results/physics_demo_images_{dimension.lower()}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_size = (12, 8)
        self.dpi = 300

    def get_next_point(self):
        """Generate a random starting point."""
        import random
        from shapely.geometry import Point

        if self.dimension == "3D":
            return Point(
                random.uniform(self.x_min + 5, self.x_max - 5),
                random.uniform(self.y_min + 5, self.y_max - 5),
                random.uniform(self.z_min + 5, self.z_max - 5),
            )
        else:
            return Point(
                random.uniform(self.x_min + 5, self.x_max - 5),
                random.uniform(self.y_min + 5, self.y_max - 5),
            )

    def get_next_velocity(self):
        """Generate random velocity component."""
        import random

        return random.uniform(-15.0, 15.0)  # m/s


def plot_trajectory(trajectory, config, filename="trajectory.png"):
    """Plot the complete trajectory with bounding box."""
    coords = list(trajectory.ls.coords)
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    fig, ax = plt.subplots(figsize=config.figure_size)

    # Plot trajectory
    ax.plot(x_coords, y_coords, "b-", linewidth=2, alpha=0.7, label="Trajectory")
    ax.scatter(
        x_coords[0],
        y_coords[0],
        color="green",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    ax.scatter(
        x_coords[-1],
        y_coords[-1],
        color="red",
        s=100,
        marker="s",
        label="End",
        zorder=5,
    )

    # Draw bounding box
    rect = patches.Rectangle(
        (config.x_min, config.y_min),
        config.x_max - config.x_min,
        config.y_max - config.y_min,
        linewidth=3,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        alpha=0.8,
    )
    ax.add_patch(rect)

    # Formatting
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_title(
        "Physics-Informed Trajectory: Bouncing Ball Simulation",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")

    # Set axis limits with padding
    padding = 2.0
    ax.set_xlim(config.x_min - padding, config.x_max + padding)
    ax.set_ylim(config.y_min - padding, config.y_max + padding)

    plt.tight_layout()
    filepath = config.output_dir / filename
    plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
    print(f"Saved trajectory plot: {filepath}")
    plt.close()

    return filepath


def plot_time_series(trajectory, config, filename="time_series.png"):
    """Plot position and velocity over time."""
    coords = list(trajectory.ls.coords)
    times = trajectory.t
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    # Calculate velocities
    vx = np.diff(x_coords) / np.diff(times)
    vy = np.diff(y_coords) / np.diff(times)
    v_times = [(times[i] + times[i + 1]) / 2 for i in range(len(times) - 1)]

    fig, axes = plt.subplots(2, 2, figsize=config.figure_size)

    # Position plots
    axes[0, 0].plot(times, x_coords, "b-", linewidth=2)
    axes[0, 0].set_title("X Position vs Time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("X Position (m)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, y_coords, "r-", linewidth=2)
    axes[0, 1].set_title("Y Position vs Time")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Y Position (m)")
    axes[0, 1].grid(True, alpha=0.3)

    # Velocity plots
    axes[1, 0].plot(v_times, vx, "b--", linewidth=2)
    axes[1, 0].set_title("X Velocity vs Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("X Velocity (m/s)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(v_times, vy, "r--", linewidth=2)
    axes[1, 1].set_title("Y Velocity vs Time")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Y Velocity (m/s)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Physics Simulation: Position and Velocity Analysis",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    filepath = config.output_dir / filename
    plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
    print(f"Saved time series plot: {filepath}")
    plt.close()

    return filepath


def plot_phase_space(trajectory, config, filename="phase_space.png"):
    """Plot phase space diagrams (position vs velocity)."""
    coords = list(trajectory.ls.coords)
    times = trajectory.t
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    # Calculate velocities
    vx = np.diff(x_coords) / np.diff(times)
    vy = np.diff(y_coords) / np.diff(times)

    fig, axes = plt.subplots(1, 2, figsize=config.figure_size)

    # X phase space
    axes[0].plot(x_coords[:-1], vx, "b-", alpha=0.7, linewidth=2)
    axes[0].scatter(
        x_coords[0],
        vx[0] if len(vx) > 0 else 0,
        color="green",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    axes[0].set_xlabel("X Position (m)")
    axes[0].set_ylabel("X Velocity (m/s)")
    axes[0].set_title("X Phase Space (Position vs Velocity)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Y phase space
    axes[1].plot(y_coords[:-1], vy, "r-", alpha=0.7, linewidth=2)
    axes[1].scatter(
        y_coords[0],
        vy[0] if len(vy) > 0 else 0,
        color="green",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    axes[1].set_xlabel("Y Position (m)")
    axes[1].set_ylabel("Y Velocity (m/s)")
    axes[1].set_title("Y Phase Space (Position vs Velocity)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(
        "Physics Simulation: Phase Space Analysis", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    filepath = config.output_dir / filename
    plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
    print(f"Saved phase space plot: {filepath}")
    plt.close()

    return filepath


def create_trajectory_animation(
    trajectory, config, filename="trajectory_animation.gif"
):
    """Create an animated visualization of the ball moving through the trajectory."""
    coords = list(trajectory.ls.coords)
    times = trajectory.t
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    fig, ax = plt.subplots(figsize=config.figure_size)

    # Set up the plot
    ax.set_xlim(config.x_min - 2, config.x_max + 2)
    ax.set_ylim(config.y_min - 2, config.y_max + 2)
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_title(
        "Physics-Informed Trajectory: Bouncing Ball Animation",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Draw bounding box
    rect = patches.Rectangle(
        (config.x_min, config.y_min),
        config.x_max - config.x_min,
        config.y_max - config.y_min,
        linewidth=3,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        alpha=0.8,
    )
    ax.add_patch(rect)

    # Initialize empty line for trajectory trail
    (trail_line,) = ax.plot([], [], "b-", alpha=0.3, linewidth=1, label="Trail")
    ball = ax.scatter([], [], s=200, c="red", marker="o", zorder=5, label="Ball")

    # Add legend
    ax.legend(fontsize=10)

    # Animation function
    def animate(frame):
        # Show trajectory trail up to current point
        if frame > 0:
            trail_line.set_data(x_coords[:frame], y_coords[:frame])

        # Update ball position
        if frame < len(x_coords):
            ball.set_offsets([(x_coords[frame], y_coords[frame])])
            ax.set_title(
                f"Physics-Informed Trajectory: t={times[frame]:.2f}s",
                fontsize=14,
                fontweight="bold",
            )

        return trail_line, ball

    # Create animation (sample every 5th frame for smoother playback)
    frame_skip = 5
    frames = range(0, len(x_coords), frame_skip)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=50, blit=False, repeat=True
    )

    # Save animation
    filepath = config.output_dir / filename
    try:
        anim.save(filepath, writer="pillow", fps=20, dpi=150)
        print(f"Saved trajectory animation: {filepath}")
    except Exception as e:
        print(
            f"Warning: Could not save animation ({e}). Try installing pillow: pip install pillow"
        )
        filepath = None

    plt.close()
    return filepath


def main():
    """Run the physics simulation demo."""
    print("Physics-Informed Trajectory Generation Demo")
    print("=" * 50)

    # Run 2D demo
    print("\n" + "=" * 20 + " 2D SIMULATION " + "=" * 20)
    run_demo_2d()

    # Run 3D demo
    print("\n" + "=" * 20 + " 3D SIMULATION " + "=" * 20)
    run_demo_3d()

    print("\nBoth 2D and 3D demos completed!")


def run_demo_2d():
    """Run the 2D physics simulation demo."""
    # Create configuration
    config = DemoConfig(dimension="2D")
    print(
        f"Bounding box: ({config.x_min}, {config.y_min}) to ({config.x_max}, {config.y_max})"
    )
    print(f"Gravity: {config.gravity} m/s²")
    print(f"Simulation time: {config.simulation_time}s")
    print(f"Time step: {config.time_step}s")
    print(f"Bounce damping: {config.bounce_damping}")

    # Create strategy
    strategy = PhysicsInformedCombinedStrategy(config)

    # Generate physics-informed trajectory
    print("\nGenerating 2D physics simulation...")
    result = strategy(trajectory_id=1)

    # Display results
    print(f"\nGenerated trajectory: {result.id}")
    print(f"Number of trajectory points: {len(result.ls.coords)}")
    print(f"Simulation duration: {result.t[-1]:.2f}s")

    # Analyze trajectory
    coords = list(result.ls.coords)
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    print(f"X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
    print(f"Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")
    print(f"Maximum height: {max(y_coords):.2f}m")

    # Count bounces (approximate)
    floor_touches = sum(1 for y in y_coords if y <= config.y_min + 0.5)
    wall_touches = sum(
        1 for x in x_coords if x <= config.x_min + 0.5 or x >= config.x_max - 0.5
    )

    print(f"Approximate floor bounces: {floor_touches}")
    print(f"Approximate wall bounces: {wall_touches}")

    # Calculate trajectory properties
    props = strategy._calculate_trajectory_properties(coords)
    print(f"Total path distance: {props['total_distance']:.2f}m")
    print(f"Height variation: {props['height_range']:.2f}m")

    # Show first few and last few points
    print("\nFirst 5 trajectory points:")
    for i in range(min(5, len(coords))):
        x, y = coords[i]
        t = result.t[i]
        print(f"  t={t:.3f}s: ({x:.2f}, {y:.2f})")

    if len(coords) > 10:
        print("\nLast 5 trajectory points:")
        for i in range(max(0, len(coords) - 5), len(coords)):
            x, y = coords[i]
            t = result.t[i]
            print(f"  t={t:.3f}s: ({x:.2f}, {y:.2f})")

    # Generate visualization images
    print("\nGenerating visualization images...")

    # Plot complete trajectory
    plot_trajectory(result, config, "bouncing_ball_trajectory.png")

    # Plot time series analysis
    plot_time_series(result, config, "position_velocity_analysis.png")

    # Plot phase space diagrams
    plot_phase_space(result, config, "phase_space_analysis.png")

    # Create trajectory animation
    print("Creating trajectory animation...")
    create_trajectory_animation(result, config, "bouncing_ball_animation.gif")

    print(f"\nAll images saved to: {config.output_dir.absolute()}")

    print("\nDemo completed!")
    print("\nGenerated files:")
    print(f"  - {config.output_dir}/bouncing_ball_trajectory.png")
    print(f"  - {config.output_dir}/position_velocity_analysis.png")
    print(f"  - {config.output_dir}/phase_space_analysis.png")
    print(f"  - {config.output_dir}/bouncing_ball_animation.gif")


def run_demo_3d():
    """Run the 3D physics simulation demo."""
    # Create configuration
    config = DemoConfig(dimension="3D")
    print(
        f"Bounding box: ({config.x_min}, {config.y_min}, {config.z_min}) to ({config.x_max}, {config.y_max}, {config.z_max})"
    )
    print(f"Gravity: {config.gravity} m/s²")
    print(f"Simulation time: {config.simulation_time}s")
    print(f"Time step: {config.time_step}s")
    print(f"Bounce damping: {config.bounce_damping}")

    # Create strategy
    strategy = PhysicsInformedCombinedStrategy(config)

    # Generate physics-informed trajectory
    print("\\nGenerating 3D physics simulation...")
    result = strategy(trajectory_id=1)

    # Display results
    print(f"\\nGenerated 3D trajectory: {result.id}")
    print(f"Number of trajectory points: {len(result.ls.coords)}")
    print(f"Simulation duration: {result.t[-1]:.2f}s")

    # Analyze trajectory
    coords = list(result.ls.coords)
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    z_coords = [c[2] for c in coords]

    print(f"X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
    print(f"Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")
    print(f"Z range: {min(z_coords):.2f} to {max(z_coords):.2f}")
    print(f"Maximum height: {max(y_coords):.2f}m")

    # Count bounces (approximate)
    floor_touches = sum(1 for y in y_coords if y <= config.y_min + 0.5)
    wall_touches = sum(
        1 for x in x_coords if x <= config.x_min + 0.5 or x >= config.x_max - 0.5
    )
    z_wall_touches = sum(
        1 for z in z_coords if z <= config.z_min + 0.5 or z >= config.z_max - 0.5
    )

    print(f"Approximate floor bounces: {floor_touches}")
    print(f"Approximate X-wall bounces: {wall_touches}")
    print(f"Approximate Z-wall bounces: {z_wall_touches}")

    # Calculate trajectory properties
    props = strategy._calculate_trajectory_properties(coords)
    print(f"Total path distance: {props['total_distance']:.2f}m")
    print(f"Height variation: {props['height_range']:.2f}m")

    # Show first few and last few points
    print("\nFirst 5 trajectory points:")
    for i in range(min(5, len(coords))):
        x, y, z = coords[i]
        t = result.t[i]
        print(f"  t={t:.3f}s: ({x:.2f}, {y:.2f}, {z:.2f})")

    if len(coords) > 10:
        print("\nLast 5 trajectory points:")
        for i in range(max(0, len(coords) - 5), len(coords)):
            x, y, z = coords[i]
            t = result.t[i]
            print(f"  t={t:.3f}s: ({x:.2f}, {y:.2f}, {z:.2f})")

    # Generate visualization images
    print("\nGenerating 3D visualization images...")

    # Plot 3D trajectory
    plot_3d_trajectory(result, config, "bouncing_ball_trajectory_3d.png")

    # Plot 3D trajectory projections
    plot_3d_projections(result, config, "trajectory_projections_3d.png")

    print(f"\n3D images saved to: {config.output_dir.absolute()}")

    print("\n3D Demo completed!")
    print("\nGenerated 3D files:")
    print(f"  - {config.output_dir}/bouncing_ball_trajectory_3d.png")
    print(f"  - {config.output_dir}/trajectory_projections_3d.png")


def plot_3d_trajectory(trajectory, config, filename="trajectory_3d.png"):
    """Plot 3D trajectory."""
    from mpl_toolkits.mplot3d import Axes3D

    coords = list(trajectory.ls.coords)
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    z_coords = [c[2] for c in coords]

    fig = plt.figure(figsize=config.figure_size)
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory
    ax.plot(
        x_coords, y_coords, z_coords, "b-", linewidth=2, alpha=0.7, label="Trajectory"
    )
    ax.scatter(
        x_coords[0],
        y_coords[0],
        z_coords[0],
        color="green",
        s=100,
        marker="o",
        label="Start",
    )
    ax.scatter(
        x_coords[-1],
        y_coords[-1],
        z_coords[-1],
        color="red",
        s=100,
        marker="s",
        label="End",
    )

    # Draw bounding box wireframe
    vertices = [
        [config.x_min, config.y_min, config.z_min],
        [config.x_max, config.y_min, config.z_min],
        [config.x_max, config.y_max, config.z_min],
        [config.x_min, config.y_max, config.z_min],
        [config.x_min, config.y_min, config.z_max],
        [config.x_max, config.y_min, config.z_max],
        [config.x_max, config.y_max, config.z_max],
        [config.x_min, config.y_max, config.z_max],
    ]

    # Define the 12 edges of the cube
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical edges
    ]

    for edge in edges:
        points = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot3D(
            [points[0][0], points[1][0]],
            [points[0][1], points[1][1]],
            [points[0][2], points[1][2]],
            "k--",
            alpha=0.6,
            linewidth=1,
        )

    # Formatting
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_zlabel("Z Position (m)", fontsize=12)
    ax.set_title(
        "Physics-Informed 3D Trajectory: Bouncing Ball Simulation",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    filepath = config.output_dir / filename
    plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
    print(f"Saved 3D trajectory plot: {filepath}")
    plt.close()

    return filepath


def plot_3d_projections(trajectory, config, filename="projections_3d.png"):
    """Plot 3D trajectory projections onto 2D planes."""
    coords = list(trajectory.ls.coords)
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    z_coords = [c[2] for c in coords]

    fig, axes = plt.subplots(2, 2, figsize=config.figure_size)

    # XY projection (top view)
    axes[0, 0].plot(x_coords, y_coords, "b-", linewidth=2, alpha=0.7)
    axes[0, 0].scatter(x_coords[0], y_coords[0], color="green", s=50, marker="o")
    axes[0, 0].scatter(x_coords[-1], y_coords[-1], color="red", s=50, marker="s")
    axes[0, 0].set_xlabel("X Position (m)")
    axes[0, 0].set_ylabel("Y Position (m)")
    axes[0, 0].set_title("XY Projection (Top View)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect("equal")

    # XZ projection (front view)
    axes[0, 1].plot(x_coords, z_coords, "r-", linewidth=2, alpha=0.7)
    axes[0, 1].scatter(x_coords[0], z_coords[0], color="green", s=50, marker="o")
    axes[0, 1].scatter(x_coords[-1], z_coords[-1], color="red", s=50, marker="s")
    axes[0, 1].set_xlabel("X Position (m)")
    axes[0, 1].set_ylabel("Z Position (m)")
    axes[0, 1].set_title("XZ Projection (Front View)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect("equal")

    # YZ projection (side view)
    axes[1, 0].plot(y_coords, z_coords, "g-", linewidth=2, alpha=0.7)
    axes[1, 0].scatter(y_coords[0], z_coords[0], color="green", s=50, marker="o")
    axes[1, 0].scatter(y_coords[-1], z_coords[-1], color="red", s=50, marker="s")
    axes[1, 0].set_xlabel("Y Position (m)")
    axes[1, 0].set_ylabel("Z Position (m)")
    axes[1, 0].set_title("YZ Projection (Side View)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect("equal")

    # Time vs Height
    times = trajectory.t
    axes[1, 1].plot(times, y_coords, "purple", linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Y Position (Height, m)")
    axes[1, 1].set_title("Height vs Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("3D Trajectory Projections", fontsize=14, fontweight="bold")
    plt.tight_layout()

    filepath = config.output_dir / filename
    plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
    print(f"Saved 3D projections plot: {filepath}")
    plt.close()

    return filepath


if __name__ == "__main__":
    main()
