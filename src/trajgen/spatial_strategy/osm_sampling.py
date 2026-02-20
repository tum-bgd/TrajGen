import random
from shapely.geometry import LineString, Point
import osmnx as ox
from pyrosm import OSM
import numpy as np
import os
import networkx as nx
import osmium
from ..config import Config
from ..trajectory import Trajectory
from .requirements_helpers import bbox_requirements


class OsmSamplingStrategy:
    """
    Generate trajectories by sampling shortest paths between public-transport
    'hotspot' nodes on an OSM road graph, adding perpendicular jitter.
    """

    config: Config
    _rng: random.Random

    def __init__(self, config: Config):
        self.config = config

        # Optional: bound total graph construction to one-time cost
        self._rng = random.Random(config.seed)
        self._G: nx.MultiDiGraph | None = None  # unprojected graph
        self._Gp: nx.MultiDiGraph | None = None  # projected graph
        self._hotspots: list[int] | None = None

        self._init_graph_and_hotspots()

    def __call__(self, id: int) -> Trajectory:
        """
        Generate a single normalized jittered OSM-based trajectory for the given id.
        """
        if self._G is None or self._Gp is None or self._hotspots is None:
            raise RuntimeError("OSM graphs or hotspots not initialized.")

        for _attempt in range(self.config.osm_max_attempts_per_traj):
            s, t = self._sample_od_nodes(self._hotspots, self._G)
            path = self._shortest_path_with_limits(self._G, s, t)
            if path is None:
                continue

            xs_m, ys_m = self._path_to_jittered_points(self._Gp, path)
            if len(xs_m) < 5:
                continue

            # Reproject to lat/lon
            line_m = LineString([Point(x, y) for x, y in zip(xs_m, ys_m)])
            line_ll, _ = ox.projection.project_geometry(
                line_m, crs=self._Gp.graph["crs"], to_crs="EPSG:4326"
            )
            if isinstance(line_ll, LineString):
                xs_ll, ys_ll = zip(*line_ll.coords)
            else:
                xs_ll, ys_ll = xs_m, ys_m

            xs_arr = np.asarray(xs_ll, dtype=float)
            ys_arr = np.asarray(ys_ll, dtype=float)

            # per-trajectory min-max normalization to [0, 1]
            xs_norm, ys_norm = self._normalize_xy(xs_arr, ys_arr)

            if len(xs_norm) == 0:
                continue

            ls = LineString(np.column_stack((xs_norm, ys_norm)))
            return Trajectory(id=id, ls=ls)

        # If we exhausted attempts, you may choose to raise or fall back.
        raise RuntimeError("Failed to generate OSM-based trajectory after retries.")

    # ------------------------------------------------------------------
    # initialization helpers
    # ------------------------------------------------------------------

    def _init_graph_and_hotspots(self) -> None:
        ox.settings.overpass_max_query_area_size = 2_000_000
        ox.settings.overpass_timeout = 300
        ox.settings.use_cache = True
        ox.settings.log_console = True  # TODO: Set to False to disable console logging
        ox.settings.log_level = "WARNING"

        if not os.path.exists(self.config.osm_pbf_path):
            raise FileNotFoundError(
                f"OSM PBF file not found: {self.config.osm_pbf_path}"
            )

        print(f"Loading drivable OSM network from {self.config.osm_pbf_path}")

        bbox = [
            self.config.x_min,
            self.config.y_min,
            self.config.x_max,
            self.config.y_max,
        ]
        print(f"Validating bbox {bbox} against PBF bounds...")
        if not self._pbf_has_bbox_coverage(
            self.config.x_min, self.config.y_min, self.config.x_max, self.config.y_max
        ):
            raise ValueError(
                f"BBox {bbox} has NO OVERLAP with PBF {self.config.osm_pbf_path}. "
                "Check coordinates or use larger bbox."
            )

        print("Loading OSM data and building road network graph...")
        osm = OSM(self.config.osm_pbf_path, bounding_box=bbox)
        print("Building OSM road network graph...")
        # Load data seperately and strip aLL extra columns immediately

        nodes, edges = osm.get_network(
            network_type="driving", nodes=True  # Returns (nodes_gdf, edges_gdf) tuple!
        )
        nodes = nodes[["id", "x", "y"]]
        edges = edges[["u", "v", "length", "geometry"]]

        self._G = osm.to_graph(nodes, edges)

        print("Projecting graph to local metric CRS")
        self._Gp = ox.project_graph(
            self._G
        )  # auto-chooses UTM-like CRS [web:6][web:15]

        print("Collecting public-transport hotspots from PBF")
        self._hotspots = self._get_pt_hotspots(self._G)

        if not self._hotspots:
            raise RuntimeError("No public transport hotspots found in OSM PBF.")

    # ------------------------------------------------------------------
    # logic lifted/adapted from your script
    # ------------------------------------------------------------------

    def _pbf_has_bbox_coverage(
        self, lon_min: float, lat_min: float, lon_max: float, lat_max: float
    ) -> bool:
        """Check if bbox intersects PBF bounds"""
        try:
            with osmium.io.Reader(self.config.osm_pbf_path) as reader:
                header = reader.header()
                bbox_header = header.box()
                print(f"PBF bounds: {bbox_header}")
                if bbox_header.valid:
                    bl = bbox_header.bottom_left  # osmium.osm.Location
                    tr = bbox_header.top_right  # osmium.osm.Location
                    h_lon_min, h_lat_min = bl.lon, bl.lat
                    h_lon_max, h_lat_max = tr.lon, tr.lat
                    return not (
                        lon_max < h_lon_min
                        or lon_min > h_lon_max
                        or lat_max < h_lat_min
                        or lat_min > h_lat_max
                    )

        except Exception as e:
            print(f"Warning: Could not validate bbox (using anyway): {e}")
            return True  # Fail-open for safety

    def _get_pt_hotspots(self, G: nx.MultiDiGraph) -> list[int]:
        osm = OSM(self.pbf_path)

        # Closely mirrors your pt_tags; implemented with pyrosm POI API. [web:7][web:13]
        pt_tags: dict[str, Any] = {
            "highway": ["bus_stop"],
            "public_transport": ["stop_position", "station"],
            "railway": ["station", "halt", "tram_stop"],
        }

        pois = osm.get_pois(custom_filter=pt_tags)

        nodes: list[int] = []
        for pt in pois.geometry.centroid:
            nearest = ox.distance.nearest_nodes(G, pt.x, pt.y)
            nodes.append(nearest)

        hotspots = list(sorted(set(nodes)))
        print("Found %d PT hotspots", len(hotspots))
        return hotspots

    def _sample_od_nodes(
        self, hotspots: list[int], G: nx.MultiDiGraph
    ) -> tuple[int, int]:
        # independent RNG for reproducibility, biased to distinct nodes
        s = self._rng.choice(hotspots)
        t = self._rng.choice(hotspots)
        while t == s:
            t = self._rng.choice(hotspots)
        return int(s), int(t)

    def _shortest_path_with_limits(
        self, G: nx.MultiDiGraph, s: int, t: int
    ) -> list[int] | None:
        try:
            path = nx.shortest_path(G, s, t, weight="length")
            length = nx.path_weight(G, path, weight="length")
            if (
                len(path) > self.config.osm_max_hops
                or length > self.config.osm_max_meters
            ):
                return None
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _jitter_point_along_segment(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        std: float,
    ) -> tuple[float, float]:
        t = self._rng.random()
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        dx = x2 - x1
        dy = y2 - y1
        norm = np.hypot(dx, dy)
        if norm == 0.0:
            return x, y

        nxp = -dy / norm
        nyp = dx / norm

        offset = self._rng.normal(0.0, std)
        xj = x + offset * nxp
        yj = y + offset * nyp
        return xj, yj

    def _path_to_jittered_points(
        self,
        Gp: nx.MultiDiGraph,
        path: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = []
        ys: list[float] = []

        for u, v in zip(path[:-1], path[1:]):
            data = min(
                Gp[u][v].values(),
                key=lambda d: d.get("length", 1.0),
            )
            geom = data.get("geometry")
            if geom is None:
                coords = [
                    (Gp.nodes[u]["x"], Gp.nodes[u]["y"]),
                    (Gp.nodes[v]["x"], Gp.nodes[v]["y"]),
                ]
            else:
                coords = list(geom.coords)

            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                n_samples = max(
                    2,
                    int(np.hypot(x2 - x1, y2 - y1) / 30.0),
                )
                for _ in range(n_samples):
                    xj, yj = self._jitter_point_along_segment(
                        x1, y1, x2, y2, self.jitter_std
                    )
                    xs.append(xj)
                    ys.append(yj)

        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    @staticmethod
    def _normalize_xy(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if xs.size == 0 or ys.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        if x_max > x_min:
            xs_norm = (xs - x_min) / (x_max - x_min)
        else:
            xs_norm = np.zeros_like(xs)

        if y_max > y_min:
            ys_norm = (ys - y_min) / (y_max - y_min)
        else:
            ys_norm = np.zeros_like(ys)

        return xs_norm, ys_norm

    @staticmethod
    def get_requirements(spatial_dim: str = "2D") -> dict:
        return {
            "get_next_osm_max_hops": {
                "short_name": "Max Hops",
                "type": "get_int_function",
                "default": 50,
                "description": "Maximum number of nodes in a sampled path.",
                "optional": False,
            },
            "get_next_osm_max_meters": {
                "short_name": "Max Meters",
                "type": "get_float_function",
                "default": 5000.0,
                "description": "Maximum path length in meters.",
                "optional": False,
            },
            "get_next_osm_max_attempts_per_traj": {
                "short_name": "Max Attempts",
                "type": "get_int_function",
                "default": 5,
                "description": "Retry attempts per trajectory.",
                "optional": True,
            },
            **bbox_requirements(spatial_dim),
        }


class _BBoxOverlapHandler(osmium.SimpleHandler):
    """Lightweight handler to detect bbox overlap."""

    def __init__(self, lon_min, lat_min, lon_max, lat_max):
        super().__init__()
        self.lon_min, self.lat_min = lon_min, lat_min
        self.lon_max, self.lat_max = lon_max, lat_max
        self.found_overlap = False
        self._stop_after_overlap = False

    def node(self, n):
        # Stop scanning after first overlap (very fast)
        if (
            self.lon_min <= n.location.lon <= self.lon_max
            and self.lat_min <= n.location.lat <= self.lat_max
        ):
            self.found_overlap = True
            self._stop_after_overlap = True
            raise osmium.Stop()
