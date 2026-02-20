import pytest
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from unittest.mock import Mock, patch
from trajgen.spatial_strategy.osm_sampling import OsmSamplingStrategy
from trajgen.config import Config
from trajgen.trajectory import Trajectory
import random


@pytest.fixture
def mock_config():
    config = Mock(spec=Config)
    config.seed = 42
    config.osm_pbf_path = "/path/to/test.osm.pbf"
    config.osm_max_attempts_per_traj = 5
    return config


@pytest.fixture
def mock_graph():
    G = nx.MultiDiGraph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=1.0, y=0.0)
    G.add_node(3, x=1.0, y=1.0)
    G.add_edge(1, 2, length=100.0)
    G.add_edge(2, 3, length=100.0)
    return G


@pytest.fixture
def mock_projected_graph():
    Gp = nx.MultiDiGraph()
    Gp.graph["crs"] = "EPSG:32633"
    Gp.add_node(1, x=100.0, y=100.0)
    Gp.add_node(2, x=200.0, y=100.0)
    Gp.add_node(3, x=200.0, y=200.0)
    Gp.add_edge(1, 2, length=100.0)
    Gp.add_edge(2, 3, length=100.0)
    return Gp


class TestOsmOnStreetNetworkStrategy:

    @patch("trajgen.spatial_strategy.osm_sampling.os.path.exists")
    @patch("trajgen.spatial_strategy.osm_sampling.OSM")
    @patch("trajgen.spatial_strategy.osm_sampling.ox.project_graph")
    def test_init_with_valid_config(
        self,
        mock_project_graph,
        mock_osm_class,
        mock_exists,
        mock_config,
        mock_graph,
        mock_projected_graph,
    ):
        mock_exists.return_value = True
        mock_osm = Mock()
        # Create mock dataframes for get_network return value (nodes_gdf, edges_gdf)
        mock_nodes = Mock()
        mock_edges = Mock()
        mock_nodes.__getitem__ = Mock(return_value=mock_nodes)  # For indexing operations
        mock_edges.__getitem__ = Mock(return_value=mock_edges)  # For indexing operations
        mock_osm.get_network.return_value = (mock_nodes, mock_edges)
        mock_osm.to_graph.return_value = mock_graph
        mock_osm.get_pois.return_value = Mock()
        mock_osm.get_pois.return_value.geometry.centroid = [Point(0.5, 0.5)]
        mock_osm._rng = random.Random(mock_config.seed)
        mock_osm_class.return_value = mock_osm
        mock_project_graph.return_value = mock_projected_graph

        with patch.object(OsmSamplingStrategy, "_get_pt_hotspots", return_value=[1, 2]):
            strategy = OsmSamplingStrategy(mock_config)

        assert strategy.config == mock_config
        assert strategy._G == mock_graph
        assert strategy._Gp == mock_projected_graph
        assert strategy._hotspots == [1, 2]

    @patch("trajgen.spatial_strategy.osm_sampling.os.path.exists")
    def test_init_missing_pbf_file(self, mock_exists, mock_config):
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="OSM PBF file not found"):
            OsmSamplingStrategy(mock_config)

    def test_normalize_xy_normal_case(self):
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([10.0, 20.0, 30.0])

        xs_norm, ys_norm = OsmSamplingStrategy._normalize_xy(xs, ys)

        assert np.allclose(xs_norm, [0.0, 0.5, 1.0])
        assert np.allclose(ys_norm, [0.0, 0.5, 1.0])

    def test_normalize_xy_empty_arrays(self):
        xs = np.array([])
        ys = np.array([])

        xs_norm, ys_norm = OsmSamplingStrategy._normalize_xy(xs, ys)
        assert xs_norm.size == 0
        assert ys_norm.size == 0

    def test_normalize_xy_constant_values(self):
        xs = np.array([5.0, 5.0, 5.0])
        ys = np.array([10.0, 10.0, 10.0])
        xs_norm, ys_norm = OsmSamplingStrategy._normalize_xy(xs, ys)

        assert np.allclose(xs_norm, [0.0, 0.0, 0.0])
        assert np.allclose(ys_norm, [0.0, 0.0, 0.0])

    def test_sample_od_nodes(self, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy._rng = Mock()
        strategy._rng.choice.side_effect = [1, 2]

        hotspots = [1, 2, 3]
        mock_graph = Mock()
        s, t = strategy._sample_od_nodes(hotspots, mock_graph)

        assert s == 1
        assert t == 2
        assert strategy._rng.choice.call_count == 2

    def test_sample_od_nodes_same_node_retry(self, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy._rng = Mock()
        strategy._rng.choice.side_effect = [
            1,
            1,
            2,
        ]  # First t same as s, then different

        hotspots = [1, 2, 3]
        mock_graph = Mock()

        s, t = strategy._sample_od_nodes(hotspots, mock_graph)

        assert s == 1
        assert t == 2
        assert strategy._rng.choice.call_count == 3

    def test_shortest_path_with_limits_success(self, mock_config, mock_graph):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        mock_config.osm_max_hops = 20
        mock_config.osm_max_meters = 2000
        strategy.config = mock_config
        strategy.max_hops = 10
        strategy.max_meters = 1000

        with patch(
            "trajgen.spatial_strategy.osm_sampling.nx.shortest_path", return_value=[1, 2]
        ):
            with patch(
                "trajgen.spatial_strategy.osm_sampling.nx.path_weight", return_value=150.0
            ):
                path = strategy._shortest_path_with_limits(mock_graph, 1, 2)

        assert path == [1, 2]

    def test_shortest_path_with_limits_too_long(self, mock_config, mock_graph):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        mock_config.osm_max_hops = 5
        mock_config.osm_max_meters = 50
        strategy.config = mock_config
        strategy.max_hops = 10
        strategy.max_meters = 100

        with patch(
            "trajgen.spatial_strategy.osm_sampling.nx.shortest_path", return_value=[1, 2]
        ):
            with patch(
                "trajgen.spatial_strategy.osm_sampling.nx.path_weight", return_value=150.0
            ):
                path = strategy._shortest_path_with_limits(mock_graph, 1, 2)

        assert path is None

    def test_shortest_path_with_limits_no_path(self, mock_config, mock_graph):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy.max_hops = 10
        strategy.max_meters = 1000

        with patch(
            "trajgen.spatial_strategy.osm_sampling.nx.shortest_path",
            side_effect=nx.NetworkXNoPath,
        ):
            path = strategy._shortest_path_with_limits(mock_graph, 1, 2)

        assert path is None

    def test_jitter_point_along_segment(self, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy._rng = Mock()
        strategy._rng.random.return_value = 0.5
        strategy._rng.normal.return_value = 10.0

        xj, yj = strategy._jitter_point_along_segment(0.0, 0.0, 100.0, 0.0, 5.0)

        # Point should be at middle of segment (50, 0) plus perpendicular offset
        assert xj == 50.0  # No perpendicular component in x
        assert yj == 10.0  # Perpendicular offset in y

    def test_jitter_point_along_segment_zero_length(self, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy._rng = Mock()
        strategy._rng.random.return_value = 0.5

        xj, yj = strategy._jitter_point_along_segment(5.0, 5.0, 5.0, 5.0, 5.0)

        assert xj == 5.0
        assert yj == 5.0

    def test_call_success(self, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy._G = Mock()
        strategy._Gp = Mock()
        strategy._Gp.graph = {"crs": "EPSG:32633"}
        strategy._hotspots = [1, 2, 3]

        # Mock all the method calls
        strategy._sample_od_nodes = Mock(return_value=(1, 2))
        strategy._shortest_path_with_limits = Mock(return_value=[1, 2])
        strategy._path_to_jittered_points = Mock(
            return_value=(
                np.array([100.0, 150.0, 200.0, 250.0, 300.0]),
                np.array([100.0, 120.0, 140.0, 160.0, 180.0]),
            )
        )

        with patch(
            "trajgen.spatial_strategy.osm_sampling.ox.projection.project_geometry"
        ) as mock_project:
            mock_linestring = Mock()
            mock_linestring.coords = [
                (0.1, 0.2),
                (0.3, 0.4),
                (0.5, 0.6),
                (0.7, 0.8),
                (0.9, 1.0),
            ]
            mock_project.return_value = (mock_linestring, None)

            trajectory = strategy(42)

        assert isinstance(trajectory, Trajectory)
        assert trajectory.id == 42
        assert isinstance(trajectory.ls, LineString)

    def test_call_uninitialized_graphs(self, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy._G = None
        strategy._Gp = None
        strategy._hotspots = None

        with pytest.raises(
            RuntimeError, match="OSM graphs or hotspots not initialized"
        ):
            strategy(42)

    def test_call_max_attempts_exceeded(self, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy._G = Mock()
        strategy._Gp = Mock()
        strategy._hotspots = [1, 2, 3]

        # Always return None for shortest path to force retries
        strategy._sample_od_nodes = Mock(return_value=(1, 2))
        strategy._shortest_path_with_limits = Mock(return_value=None)

        with pytest.raises(
            RuntimeError, match="Failed to generate OSM-based trajectory after retries"
        ):
            strategy(42)

    def test_path_to_jittered_points(self, mock_config, mock_projected_graph):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy.jitter_std = 5.0
        strategy._jitter_point_along_segment = Mock(
            side_effect=lambda x1, y1, x2, y2, std: (
                x1 + (x2 - x1) / 2,
                y1 + (y2 - y1) / 2,
            )
        )

        path = [1, 2]
        xs, ys = strategy._path_to_jittered_points(mock_projected_graph, path)

        assert len(xs) > 0
        assert len(ys) > 0
        assert len(xs) == len(ys)

    @patch("trajgen.spatial_strategy.osm_sampling.ox.distance.nearest_nodes")
    def test_get_pt_hotspots(self, mock_nearest_nodes, mock_config):
        strategy = OsmSamplingStrategy.__new__(OsmSamplingStrategy)
        strategy.config = mock_config
        strategy.pbf_path = "/path/to/test.osm.pbf"

        mock_osm = Mock()
        mock_pois = Mock()
        mock_pois.geometry.centroid = [Point(0.5, 0.5), Point(1.5, 1.5)]
        mock_osm.get_pois.return_value = mock_pois
        mock_nearest_nodes.side_effect = [1, 2]

        with patch("trajgen.spatial_strategy.osm_sampling.OSM", return_value=mock_osm):
            hotspots = strategy._get_pt_hotspots(Mock())

        assert hotspots == [1, 2]
        assert mock_nearest_nodes.call_count == 2
