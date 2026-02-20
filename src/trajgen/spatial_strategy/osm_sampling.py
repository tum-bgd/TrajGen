import random
from typing import Any

from shapely.geometry import LineString, Point
import osmnx as ox
import numpy as np
import networkx as nx

from ..config import Config
from ..trajectory import Trajectory


class OsmSamplingStrategy:
    """
    Generate trajectories by sampling shortest paths between public-transport
    'hotspot' nodes on an OSM road graph, adding perpendicular jitter.

    The road graph is fetched live from the Overpass API for the bounding box
    configured in the UI (no local PBF file required).
    """

    config: Config
    _rng: random.Random

    def __init__(self, config: Config, progress_callback=None):
        self.config = config
        # Optional callable(pct: int, msg: str) for UI progress reporting
        self._progress_callback = progress_callback

        self._rng = random.Random(config.seed)
        self._G: nx.MultiDiGraph | None = None  # unprojected graph
        self._Gp: nx.MultiDiGraph | None = None  # projected graph
        self._hotspots: list[int] | None = None

        self._init_graph_and_hotspots()

    def __call__(self, id: int) -> Trajectory:
        """
        Generate a jittered OSM-based trajectory for the given id.

        Coordinates are in geographic (lon, lat) space — not normalised —
        so they can be displayed directly on a map.
        """
        if self._G is None or self._Gp is None or self._hotspots is None:
            raise RuntimeError("OSM graphs or hotspots not initialized.")

        # Reseed per trajectory so each ID is deterministic regardless of
        # call order and independent of construction-time RNG state.
        self._rng = random.Random(self.config.seed + id)

        for _attempt in range(self.config.osm_max_attempts_per_traj):
            s, t = self._sample_od_nodes(self._hotspots, self._G)
            path = self._shortest_path_with_limits(self._G, s, t)
            if path is None:
                continue

            xs_m, ys_m = self._path_to_points(self._Gp, path)
            if len(xs_m) < 5:
                continue

            # Reproject metric coords back to lon/lat
            line_m = LineString([Point(x, y) for x, y in zip(xs_m, ys_m)])
            line_ll, _ = ox.projection.project_geometry(
                line_m, crs=self._Gp.graph["crs"], to_crs="EPSG:4326"
            )
            if not isinstance(line_ll, LineString) or len(line_ll.coords) == 0:
                continue

            return Trajectory(id=id, ls=line_ll)

        raise RuntimeError("Failed to generate OSM-based trajectory after retries.")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _report(self, pct: int, msg: str) -> None:
        print(msg)
        if self._progress_callback:
            self._progress_callback(pct, msg)

    def _init_graph_and_hotspots(self) -> None:
        ox.settings.use_cache = True
        ox.settings.log_console = False

        # bbox = (lon_min, lat_min, lon_max, lat_max) per osmnx 2.x convention
        # config stores x = longitude, y = latitude
        bbox = (
            self.config.x_min,
            self.config.y_min,
            self.config.x_max,
            self.config.y_max,
        )

        self._report(10, f"Downloading drivable OSM network for bbox {bbox} ...")
        self._G = ox.graph_from_bbox(bbox, network_type="drive")

        self._report(60, "Projecting graph to local metric CRS ...")
        self._Gp = ox.project_graph(self._G)

        self._report(75, "Collecting public-transport hotspots ...")
        self._hotspots = self._get_pt_hotspots(self._Gp, bbox)

        self._report(100, f"Done — {len(self._hotspots)} PT hotspots found.")

        if not self._hotspots:
            raise RuntimeError(
                "No public transport hotspots found in the selected area."
            )

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _get_pt_hotspots(self, Gp: nx.MultiDiGraph, bbox: tuple) -> list[int]:
        pt_tags: dict[str, Any] = {
            "highway": ["bus_stop"],
            "public_transport": ["stop_position", "station"],
            "railway": ["station", "halt", "tram_stop"],
        }
        pois = ox.features_from_bbox(bbox, tags=pt_tags)
        # Project POI geometries to match Gp so scipy KDTree is used instead
        # of the scikit-learn Ball Tree (which is required for unprojected graphs)
        pois_proj = pois.to_crs(Gp.graph["crs"])
        nodes: list[int] = []
        for pt in pois_proj.geometry.centroid:
            nearest = ox.distance.nearest_nodes(Gp, pt.x, pt.y)
            nodes.append(nearest)
        hotspots = list(sorted(set(nodes)))
        print(f"Found {len(hotspots)} PT hotspots")
        return hotspots

    def _sample_od_nodes(
        self, hotspots: list[int], G: nx.MultiDiGraph
    ) -> tuple[int, int]:
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

    def _path_to_points(
        self,
        Gp: nx.MultiDiGraph,
        path: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect the exact OSM road-geometry vertices along the path."""
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

            # Exclude the last vertex of each edge to avoid duplicating
            # the junction node shared with the next edge.
            xs.extend(c[0] for c in coords[:-1])
            ys.extend(c[1] for c in coords[:-1])

        # Append the final endpoint.
        last = path[-1]
        xs.append(Gp.nodes[last]["x"])
        ys.append(Gp.nodes[last]["y"])

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
                "default_mode": "fixed for dataset",
                "description": "Maximum number of nodes in a sampled path.",
                "optional": False,
            },
            "get_next_osm_max_meters": {
                "short_name": "Max Meters",
                "type": "get_float_function",
                "default": 5000.0,
                "default_mode": "fixed for dataset",
                "description": "Maximum path length in meters.",
                "optional": False,
            },
            "get_next_osm_max_attempts_per_traj": {
                "short_name": "Max Attempts",
                "type": "get_int_function",
                "default": 5,
                "default_mode": "fixed for dataset",
                "description": "Retry attempts per trajectory.",
                "optional": True,
            },
            # Bbox — same keys as bbox_requirements but with lat/lon defaults and labels
            "get_next_x_min": {
                "short_name": "Longitude Min",
                "type": "get_float_function",
                "default": 11.54,
                "default_mode": "fixed for dataset",
                "description": "Western boundary of the area (longitude, e.g. 11.54 for Munich).",
                "optional": False,
            },
            "get_next_x_max": {
                "short_name": "Longitude Max",
                "type": "get_float_function",
                "default": 11.62,
                "default_mode": "fixed for dataset",
                "description": "Eastern boundary of the area (longitude).",
                "optional": False,
            },
            "get_next_y_min": {
                "short_name": "Latitude Min",
                "type": "get_float_function",
                "default": 48.12,
                "default_mode": "fixed for dataset",
                "description": "Southern boundary of the area (latitude, e.g. 48.12 for Munich).",
                "optional": False,
            },
            "get_next_y_max": {
                "short_name": "Latitude Max",
                "type": "get_float_function",
                "default": 48.17,
                "default_mode": "fixed for dataset",
                "description": "Northern boundary of the area (latitude).",
                "optional": False,
            },
        }
