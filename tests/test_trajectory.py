import pytest  # noqa F401
from shapely.geometry import LineString
from trajgen.trajectory import Trajectory, TrajectoryDataset
from trajgen.config import Config


class TestTrajectory:
    def test_trajectory_creation(self):
        coords = [(0, 0), (1, 1), (2, 2)]
        ls = LineString(coords)
        traj = Trajectory(id=1, ls=ls)

        assert traj.id == 1
        assert traj.ls == ls

    def test_trajectory_len(self):
        coords = [(0, 0), (1, 1), (2, 2)]
        ls = LineString(coords)
        traj = Trajectory(id=1, ls=ls)

        assert len(traj) == 3

    def test_trajectory_getitem(self):
        coords = [(0, 0), (1, 1), (2, 2)]
        ls = LineString(coords)
        traj = Trajectory(id=1, ls=ls)

        assert traj[0] == (0, 0)
        assert traj[1] == (1, 1)
        assert traj[2] == (2, 2)

    def test_trajectory_iter(self):
        coords = [(0, 0), (1, 1), (2, 2)]
        ls = LineString(coords)
        traj = Trajectory(id=1, ls=ls)

        result = list(traj)
        assert result == coords

    def test_trajectory_repr(self):
        coords = [(0, 0), (1, 1)]
        ls = LineString(coords)
        traj = Trajectory(id=1, ls=ls)

        expected = "Trajectory(id=1, points=[(0.0, 0.0), (1.0, 1.0)])"
        assert repr(traj) == expected


class TestTrajectoryDataset:
    def test_dataset_creation(self):
        config = Config({})
        trajectories = []
        dataset = TrajectoryDataset(generator_config=config, trajectories=trajectories)

        assert dataset.generator_config == config
        assert dataset.trajectories == trajectories

    def test_dataset_len(self):
        config = Config({})
        coords1 = [(0, 0), (1, 1)]
        coords2 = [(2, 2), (3, 3)]
        traj1 = Trajectory(id=1, ls=LineString(coords1))
        traj2 = Trajectory(id=2, ls=LineString(coords2))
        trajectories = [traj1, traj2]
        dataset = TrajectoryDataset(generator_config=config, trajectories=trajectories)

        assert len(dataset) == 2

    def test_dataset_getitem(self):
        config = Config({})
        coords = [(0, 0), (1, 1)]
        traj = Trajectory(id=1, ls=LineString(coords))
        trajectories = [traj]
        dataset = TrajectoryDataset(generator_config=config, trajectories=trajectories)

        assert dataset[0] == traj

    def test_dataset_iter(self):
        config = Config({})
        coords1 = [(0, 0), (1, 1)]
        coords2 = [(2, 2), (3, 3)]
        traj1 = Trajectory(id=1, ls=LineString(coords1))
        traj2 = Trajectory(id=2, ls=LineString(coords2))
        trajectories = [traj1, traj2]
        dataset = TrajectoryDataset(generator_config=config, trajectories=trajectories)

        result = list(dataset)
        assert result == trajectories

    def test_dataset_repr(self):
        config = Config({})
        coords = [(0, 0), (1, 1)]
        traj = Trajectory(id=1, ls=LineString(coords))
        trajectories = [traj]
        dataset = TrajectoryDataset(generator_config=config, trajectories=trajectories)

        expected = "TrajectoryDataset(num_trajectories=1)"
        assert repr(dataset) == expected

    def test_dataset_save_and_load(self):
        config = Config({})
        coords1 = [(0, 0), (1, 1)]
        coords2 = [(2, 2), (3, 3)]
        traj1 = Trajectory(id=1, ls=LineString(coords1))
        traj2 = Trajectory(id=2, ls=LineString(coords2))
        trajectories = [traj1, traj2]
        dataset = TrajectoryDataset(generator_config=config, trajectories=trajectories)

        file_path = "./tests/tmp/dataset.txt"
        dataset.save(str(file_path))

        loaded_dataset = TrajectoryDataset.load(str(file_path))

        assert len(loaded_dataset) == len(dataset)
        for original, loaded in zip(dataset.trajectories, loaded_dataset.trajectories):
            assert original.id == loaded.id
            assert list(original.ls.coords) == list(loaded.ls.coords)
