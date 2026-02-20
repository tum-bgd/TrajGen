import pytest
from unittest.mock import Mock, patch
from trajgen.trajectory import Trajectory
from trajgen.spatial_strategy import SpatialStrategy
from trajgen.config import Config
from trajgen.trajectory_generator import (
    TrajectoryDatasetFileIterator,
    TrajectoryGenerator,
    TrajectoryDatasetGenerator,
)


@pytest.fixture
def mock_config():
    return Mock(spec=Config, num_trajectories=10, grid_rows=5, grid_cols=5)


@pytest.fixture
def mock_strategy():
    mock = Mock(spec=SpatialStrategy)
    mock.return_value = Mock(spec=Trajectory, id=0)
    return mock


@pytest.fixture
def mock_trajectory_generator(mock_config, mock_strategy):
    return TrajectoryGenerator(mock_config, mock_strategy)


@pytest.fixture
def mock_dataset_generator(mock_trajectory_generator, mock_config):
    return TrajectoryDatasetGenerator(mock_trajectory_generator, mock_config)


class TestTrajectoryGenerator:
    def test_init_stores_config_and_strategy(self, mock_config, mock_strategy):
        generator = TrajectoryGenerator(mock_config, mock_strategy)
        assert generator.config == mock_config
        assert generator.spatial_strategy == mock_strategy

    def test_generate_trajectory_calls_spatial_strategy(self, mock_trajectory_generator):
        mock_trajectory = Mock(spec=Trajectory)
        mock_trajectory_generator.spatial_strategy.return_value = mock_trajectory

        result = mock_trajectory_generator.generate_trajectory(42)

        mock_trajectory_generator.spatial_strategy.assert_called_once_with(
            42, mock_trajectory_generator.config
        )
        assert result == mock_trajectory


class TestTrajectoryDatasetGenerator:
    def test_init_stores_dependencies(self, mock_trajectory_generator, mock_config):
        generator = TrajectoryDatasetGenerator(mock_trajectory_generator, mock_config)
        assert generator.trajectory_generator == mock_trajectory_generator
        assert generator.config == mock_config

    def test_generate_dataset_correct_count(self, mock_dataset_generator):
        with patch.object(
            mock_dataset_generator.trajectory_generator, "generate_trajectory"
        ) as mock_generate:
            mock_generate.side_effect = lambda id: Mock(spec=Trajectory, id=id)
            dataset = mock_dataset_generator.generate_dataset(10)
            mock_generate.assert_called()

            assert len(dataset) == 10
            assert all(t.id == i for i, t in enumerate(dataset.trajectories))


class TrajectoryGeneratorFileIteratorTests:
    def test_file_iterator_reads_trajectories(self, tmp_path):
        file_content = """1: [(0.0, 0.0), (1.0, 1.0)]\n2: [(2.0, 2.0), (3.0, 3.0)]"""
        file_path = "./tests/tmp/trajectories.txt"
        with open(file_path, "w") as f:
            f.write(file_content)
        iterator = TrajectoryDatasetFileIterator(file_path)

        with iterator as it:
            trajectories = list(it)
        assert len(trajectories) == 2
        assert trajectories[0].id == 1
        assert list(trajectories[0].ls.coords) == [(0.0, 0.0), (1.0, 1.0)]
        assert trajectories[1].id == 2
        assert list(trajectories[1].ls.coords) == [(2.0, 2.0), (3.0, 3.0)]

    def test_file_iterator_skips_invalid_lines(self, tmp_path):
        file_content = (
            """1: [(0.0, 0.0), (1.0, 1.0)]\nInvalid Line\n2: [(2.0, 2.0), (3.0, 3.0)]"""
        )
        file_path = "./tests/tmp/trajectories_invalid.txt"
        with open(file_path, "w") as f:
            f.write(file_content)
        iterator = TrajectoryDatasetFileIterator(file_path)

        with iterator as it:
            trajectories = list(it)
        assert len(trajectories) == 2
        assert trajectories[0].id == 1
        assert list(trajectories[0].ls.coords) == [(0.0, 0.0), (1.0, 1.0)]
        assert trajectories[1].id == 2
        assert list(trajectories[1].ls.coords) == [(2.0, 2.0), (3.0, 3.0)]
