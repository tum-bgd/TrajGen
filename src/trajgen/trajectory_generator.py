from collections.abc import Iterator
from shapely import LineString
from .trajectory import Trajectory, TrajectoryDataset
from .spatial_strategy import SpatialStrategy
from .temporal_strategy import TemporalStrategy
from .combined_strategy import CombinedStrategy
from .resampling_strategy import ResamplingStrategy
from .config import Config


class TrajectoryGenerator:
    # config: Config
    # spatial_strategy: SpatialStrategy
    # temporal_strategy: None | TemporalStrategy
    # combined_strategy: None | CombinedStrategy
    # resampling_strategy: None | ResamplingStrategy

    def __init__(
        self,
        config: Config,
        spatial_strategy: SpatialStrategy,
        temporal_strategy: None | TemporalStrategy = None,
        combined_strategy: None | CombinedStrategy = None,
        resampling_strategy: None | ResamplingStrategy = None,
    ):
        self.config = config
        self.spatial_strategy = spatial_strategy
        self.temporal_strategy = temporal_strategy
        self.combined_strategy = combined_strategy
        self.resampling_strategy = resampling_strategy

    def generate_trajectory(self, id) -> Trajectory:
        if self.spatial_strategy:
            trajectory = self.spatial_strategy(id)
            if self.temporal_strategy:
                trajectory = self.temporal_strategy(trajectory)
        elif self.combined_strategy:
            trajectory = self.combined_strategy(id)
        if self.resampling_strategy:
            trajectory = self.resampling_strategy(trajectory)
        return trajectory


class TrajectoryDatasetGenerator:
    def __init__(self, trajectory_generator: TrajectoryGenerator, config: Config):
        self.trajectory_generator = trajectory_generator
        self.config = config

    def generate_dataset(self, num_trajectories: int) -> TrajectoryDataset:
        dataset = []
        for i in range(num_trajectories):
            trajectory = self.trajectory_generator.generate_trajectory(id=i)
            dataset.append(trajectory)
        return TrajectoryDataset(generator_config=self.config, trajectories=dataset)


class TrajectoryDatasetFileIterator:
    """Lazy iterator over trajectory file - reads line-by-line."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None

    def __enter__(self):
        self.file = open(self.filepath, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def __iter__(self) -> Iterator[Trajectory]:
        for line_num, line in enumerate(self.file, 1):
            line = line.strip()
            if not line:
                continue
            # Parse line like "1: [(0.0, 0.0), (1.0, 1.0)]"
            try:
                id_str, points_str = line.split(":", 1)
                traj_id = int(id_str.strip())

                # Extract points:
                points_str = points_str.strip()[1:-1]  # Remove outer []
                if not points_str:
                    continue

                points = eval(
                    points_str
                )  # Safe for simple tuples; use ast.literal_eval in prod
                yield Trajectory(id=traj_id, ls=LineString(points))

            except (ValueError, IndexError, SyntaxError) as e:
                print(f"Warning: Skipping invalid line {line_num}: {line} ({e})")
