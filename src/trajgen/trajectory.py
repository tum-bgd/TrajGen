from dataclasses import dataclass
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from .config import Config


@dataclass
class Trajectory:
    id: int
    ls: LineString
    t: list[float] = None  # Optional time component

    def __len__(self):
        return len(self.ls.coords)

    def __getitem__(self, index: int) -> tuple[float, float]:
        return self.ls.coords[index]

    def get_id(self):
        return self.id

    def get_time(self, index: int) -> float | None:
        return self.t[index] if self.t is not None else None
    
    def set_time(self, t: list[float]):
        self.t = t

    def __iter__(self):
        return iter(self.ls.coords)

    def __repr__(self):
        return f"Trajectory(id={self.id}, points={list(self.ls.coords)})"


@dataclass
class TrajectoryDataset:
    generator_config: Config
    trajectories: list[Trajectory]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, id):
        return self.trajectories[id]

    def __iter__(self):
        return iter(self.trajectories)

    def __repr__(self):
        return f"TrajectoryDataset(num_trajectories={len(self.trajectories)})"

    def save(self, filepath: str):
        """Saves the dataset to a file."""
        with open(filepath, "w") as f:
            for traj in self.trajectories:
                f.write(f"{traj.id}: {list(traj.ls.coords)}\n")

    @classmethod
    def load(cls, filepath: str) -> "TrajectoryDataset":
        """Loads a dataset from a file."""
        trajectories = []
        with open(filepath, "r") as f:
            for line in f:
                id_str, coords_str = line.strip().split(": ")
                id = int(id_str)
                coords = eval(coords_str)
                ls = LineString(coords)
                trajectories.append(Trajectory(id=id, ls=ls))
        return cls(generator_config=None, trajectories=trajectories)

    def visualize(self, n=10):
        """Visualizes the top n trajectories using matplotlib."""

        plt.figure(figsize=(8, 8))
        for traj in self.trajectories:
            x, y = traj.ls.xy
            plt.plot(x, y, marker="o", label=f"Trajectory {traj.id}")
        plt.title("Trajectory Dataset Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid()
        plt.show()
