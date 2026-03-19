#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""
Convert a single nuScenes scene to an alpasim-compatible .usdz artifact.

Usage (from repo root):
    conda run -n alpasim_env python src/tools/scripts/nuscenes_to_usdz.py \
        --dataroot /workspace/vla-test/alpasim/data \
        --scene-index 0 \
        --output-dir /workspace/vla-test/alpasim/data/nre-artifacts/all-usdzs

The script produces:
  - <uuid>.usdz  containing all required alpasim files
  - An appended row in data/scenes/sim_scenes.csv

Requires (in alpasim_env):
  nuscenes-devkit, open3d, pyproj, scipy, numpy, pyyaml
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from nuscenes.nuscenes import NuScenes
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────────────────────────────
# nuScenes category → alpasim label_class
# None = skip (static objects / unmapped)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_MAP: dict[str, str | None] = {
    "vehicle.car":                              "automobile",
    "vehicle.truck":                            "heavy_truck",
    "vehicle.bus.bendy":                        "heavy_truck",
    "vehicle.bus.rigid":                        "heavy_truck",
    "vehicle.trailer":                          "trailer",
    "vehicle.motorcycle":                       "automobile",
    "vehicle.bicycle":                          "bicycle",
    "vehicle.emergency.police":                 "automobile",
    "vehicle.emergency.ambulance":              "automobile",
    "vehicle.construction":                     "heavy_truck",
    "human.pedestrian.adult":                   "person",
    "human.pedestrian.child":                   "person",
    "human.pedestrian.wheelchair":              "person",
    "human.pedestrian.stroller":                "person",
    "human.pedestrian.personal_mobility":       "person",
    "human.pedestrian.police_officer":          "person",
    "human.pedestrian.construction_worker":     "person",
    "animal":                                   "animal",
    # static / non-dynamic — skip
    "movable_object.barrier":                   None,
    "movable_object.trafficcone":               None,
    "movable_object.pushable_pullable":         None,
    "movable_object.debris":                    None,
    "static_object.bicycle_rack":               None,
}

# nuScenes recording location → approximate (lat_deg, lon_deg) for ECEF transform
LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "singapore-onenorth":       (1.2902,  103.8417),
    "singapore-hollandvillage": (1.3071,  103.8088),
    "singapore-queenstown":     (1.2911,  103.8245),
    "singapore-expressway":     (1.3222,  103.7959),
    "boston-seaport":           (42.3368, -71.0499),
}

# Minimal USD stubs (NRE sensor-sim reads these; not needed for trajectory-only sim)
MINIMAL_USDA = (
    '#usda 1.0\n(\n    defaultPrim = "World"\n)\n'
    'def Xform "World"\n{\n}\n'
)
DOME_LIGHT_USDA = (
    '#usda 1.0\n(\n    defaultPrim = "DomeLight"\n)\n'
    'def DomeLight "DomeLight"\n{\n    float inputs:intensity = 1.0\n}\n'
)
MESH_USD_STUB = '#usda 1.0\ndef Mesh "{name}"\n{{\n}}\n'


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def nusc_wxyz_to_xyzw(q_wxyz: list[float]) -> list[float]:
    """nuScenes [w,x,y,z] → scipy/alpasim [x,y,z,w]."""
    w, x, y, z = q_wxyz
    return [x, y, z, w]


def make_T_world_rig(translation: list[float], rotation_wxyz: list[float]) -> np.ndarray:
    """4x4 T_world_rig from nuScenes ego_pose translation + rotation [w,x,y,z]."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(nusc_wxyz_to_xyzw(rotation_wxyz)).as_matrix()
    T[:3, 3] = translation
    return T


def make_T_rig_world(translation: list[float], rotation_wxyz: list[float]) -> np.ndarray:
    """4x4 T_rig_world = inverse of T_world_rig."""
    return np.linalg.inv(make_T_world_rig(translation, rotation_wxyz))


def compute_T_world_base(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> list:
    """
    Build a 4x4 ECEF→local-ENU matrix centred at (lat_deg, lon_deg, alt_m).
    Stored as T_world_base in rig_trajectories.json; used by alpasim to align
    the XODR road map coordinate frame to the simulation world frame.
    """
    tf = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x0, y0, z0 = tf.transform(lon_deg, lat_deg, alt_m)

    sin_lat = np.sin(np.radians(lat_deg))
    cos_lat = np.cos(np.radians(lat_deg))
    sin_lon = np.sin(np.radians(lon_deg))
    cos_lon = np.cos(np.radians(lon_deg))

    Rot = np.array([
        [-sin_lon,              cos_lon,            0.0     ],
        [-sin_lat * cos_lon,   -sin_lat * sin_lon,  cos_lat ],
        [ cos_lat * cos_lon,    cos_lat * sin_lon,  sin_lat ],
    ])

    T = np.eye(4)
    T[:3, :3] = Rot
    T[:3, 3]  = -Rot @ np.array([x0, y0, z0])
    return T.tolist()


def transform_points(pts_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) array."""
    pts_h = np.hstack([pts_xyz, np.ones((len(pts_xyz), 1))])
    return (T @ pts_h.T).T[:, :3]


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR mesh
# ─────────────────────────────────────────────────────────────────────────────

def load_lidar_bin(path: str) -> np.ndarray:
    """Load nuScenes LIDAR_TOP .pcd.bin → (N, 4) float32 [x, y, z, intensity]."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]


def build_ground_mesh_for_scene(
    nusc: NuScenes,
    scene: dict,
    dataroot: str,
    voxel_size: float = 0.3,
    ground_z_offset: float = 0.5,
) -> o3d.geometry.TriangleMesh:
    """
    Aggregate all LIDAR_TOP keyframe samples for the scene into a world-frame
    point cloud, filter to ground-level points, and run Poisson reconstruction.
    """
    print("  Loading LiDAR samples...")
    all_points: list[np.ndarray] = []
    n_sweeps = 0

    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        bin_path = Path(dataroot) / sd["filename"]

        if not bin_path.exists():
            print(f"    Missing: {bin_path}")
            sample_token = sample["next"]
            continue

        pts = load_lidar_bin(str(bin_path))[:, :3]

        # sensor frame → rig frame → world frame
        cs  = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep  = nusc.get("ego_pose",           sd["ego_pose_token"])
        T_rig_sensor  = make_T_world_rig(cs["translation"], cs["rotation"])
        T_world_rig   = make_T_world_rig(ep["translation"], ep["rotation"])
        T_world_sensor = T_world_rig @ T_rig_sensor

        all_points.append(transform_points(pts, T_world_sensor))
        n_sweeps += 1
        sample_token = sample["next"]

    raw_pts = np.vstack(all_points)
    print(f"  {n_sweeps} sweeps → {len(raw_pts):,} raw points")

    # Downsample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_pts)
    pcd = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd.points)
    print(f"  After voxel down-sample ({voxel_size} m): {len(pts):,} points")

    # Ground filter: keep points within `ground_z_offset` m above the 5th-percentile height
    z_min = np.percentile(pts[:, 2], 5)
    ground_mask = pts[:, 2] < (z_min + ground_z_offset)
    ground_pts = pts[ground_mask]
    print(f"  Ground points (z < {z_min + ground_z_offset:.2f} m): {len(ground_pts):,}")

    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(ground_pts)
    pcd_ground.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30)
    )
    pcd_ground.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

    print("  Running Poisson surface reconstruction...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_ground, depth=8, width=0, scale=1.1, linear_fit=False
    )
    # Crop to avoid artefacts at the boundary
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=ground_pts.min(axis=0) - 2.0,
        max_bound=ground_pts.max(axis=0) + 2.0,
    )
    mesh = mesh.crop(bbox)
    mesh = mesh.simplify_quadric_decimation(200_000)
    mesh.compute_vertex_normals()
    print(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh


def mesh_to_ply_bytes(mesh: o3d.geometry.TriangleMesh) -> bytes:
    """Serialise an Open3D mesh to PLY bytes without a persistent temp file."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    try:
        o3d.io.write_triangle_mesh(tmp, mesh, write_ascii=False)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# rig_trajectories.json
# ─────────────────────────────────────────────────────────────────────────────

CAM_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def build_rig_trajectories(
    nusc: NuScenes, scene: dict, sequence_id: str
) -> tuple[dict, list[int]]:
    """Return (rig_trajectories_dict, ego_timestamps_us)."""

    ego_timestamps_us: list[int] = []
    T_rig_worlds: list[list] = []
    cam_timestamps: dict[str, list[int]] = {ch: [] for ch in CAM_CHANNELS}

    # ── walk all keyframe samples ──────────────────────────────────────
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)

        # Use LIDAR_TOP ego pose as the canonical rig pose
        sd_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ep = nusc.get("ego_pose", sd_lidar["ego_pose_token"])
        ego_timestamps_us.append(ep["timestamp"])
        T_rig_worlds.append(
            make_T_rig_world(ep["translation"], ep["rotation"]).tolist()
        )

        for ch in CAM_CHANNELS:
            if ch in sample["data"]:
                sd_cam = nusc.get("sample_data", sample["data"][ch])
                cam_timestamps[ch].append(sd_cam["timestamp"])

        sample_token = sample["next"]

    # ── camera calibrations (static per scene) ────────────────────────
    first_sample = nusc.get("sample", scene["first_sample_token"])
    camera_calibrations: dict[str, dict] = {}

    for ch in CAM_CHANNELS:
        if ch not in first_sample["data"]:
            continue
        sd_cam = nusc.get("sample_data", first_sample["data"][ch])
        cs     = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])

        # T_sensor_rig: takes a point from rig frame into sensor frame
        T_rig_sensor  = make_T_world_rig(cs["translation"], cs["rotation"])  # sensor pose in rig
        T_sensor_rig  = np.linalg.inv(T_rig_sensor).tolist()

        K  = cs["camera_intrinsic"]   # [[fx,0,cx],[0,fy,cy],[0,0,1]]
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]

        cam_key = f"{ch}@{sequence_id}"
        camera_calibrations[cam_key] = {
            "sequence_id":         sequence_id,
            "logical_sensor_name": ch,
            "unique_sensor_idx":   CAM_CHANNELS.index(ch),
            "T_sensor_rig":        T_sensor_rig,
            "camera_model": {
                "type": "opencv_pinhole",
                "parameters": {
                    "resolution":      [sd_cam["width"], sd_cam["height"]],
                    "shutter_type":    "ROLLING_TOP_TO_BOTTOM",
                    "focal_length":    [fx, fy],
                    "principal_point": [cx, cy],
                    "radial":          [],
                    "tangential":      [],
                    "thin_prism":      [],
                },
            },
        }

    # ── T_world_base from recording location ──────────────────────────
    log      = nusc.get("log", scene["log_token"])
    location = log.get("location", "singapore-onenorth")
    lat, lon = LOCATION_COORDS.get(location, (1.29, 103.84))
    print(f"  Recording location: {location} → lat={lat}, lon={lon}")
    T_world_base = compute_T_world_base(lat, lon)

    rig_json = {
        "T_world_base": T_world_base,
        "world_to_nre":          {"matrix": np.eye(4).tolist()},
        "camera_calibrations":   camera_calibrations,
        "lidar_calibrations":    {},
        "rig_trajectories": [
            {
                "sequence_id": sequence_id,
                # Ford Fusion (nuScenes ego vehicle) approximate bounding box
                # Convention: centroid and dim relative to rear-axle-centre rig origin
                "rig_bbox": {
                    "centroid": [1.50, 0.0, 0.75],
                    "dim":      [4.60, 1.90, 1.50],
                    "rot":      [0.0, 0.0, 0.0],
                },
                "T_rig_worlds":               T_rig_worlds,
                "T_rig_world_timestamps_us":  ego_timestamps_us,
                "cameras_frame_timestamps_us": {
                    f"{ch}@{sequence_id}": ts
                    for ch, ts in cam_timestamps.items()
                    if ts
                },
                "lidars_frame_timestamps_us":           {},
                "cameras_linear_start_frame_indices":   {},
                "lidars_linear_start_frame_indices":    {},
            }
        ],
    }
    return rig_json, ego_timestamps_us


# ─────────────────────────────────────────────────────────────────────────────
# sequence_tracks.json
# ─────────────────────────────────────────────────────────────────────────────

def _map_category(category_name: str) -> str | None:
    """Map a nuScenes category string to an alpasim label class."""
    for prefix, label in CATEGORY_MAP.items():
        if category_name.startswith(prefix):
            return label
    return None


def build_sequence_tracks(
    nusc: NuScenes, scene: dict, sequence_id: str
) -> dict:
    """Return sequence_tracks.json dict keyed by sequence_id."""

    # instance_token → accumulated data
    instances: dict[str, dict] = {}

    sample_token = scene["first_sample_token"]
    while sample_token:
        sample   = nusc.get("sample", sample_token)
        sd_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ep       = nusc.get("ego_pose", sd_lidar["ego_pose_token"])
        ts_us    = ep["timestamp"]

        for ann_token in sample["anns"]:
            ann      = nusc.get("sample_annotation", ann_token)
            label    = _map_category(ann["category_name"])
            if label is None:
                continue

            itoken = ann["instance_token"]
            if itoken not in instances:
                w, length, h = ann["size"]   # nuScenes: [width, length, height]
                instances[itoken] = {
                    "label": label,
                    "dim":   [length, w, h],  # alpasim: [length, width, height]
                    "timestamps": [],
                    "poses":      [],
                }

            # nuScenes rotation is [w, x, y, z]; convert to scipy/alpasim [x, y, z, w]
            qx, qy, qz, qw = nusc_wxyz_to_xyzw(ann["rotation"])
            x, y, z = ann["translation"]
            instances[itoken]["timestamps"].append(ts_us)
            instances[itoken]["poses"].append([x, y, z, qx, qy, qz, qw])

        sample_token = sample["next"]

    # Keep only actors seen in ≥ 2 frames (need a valid trajectory)
    instances = {k: v for k, v in instances.items() if len(v["timestamps"]) >= 2}

    tracks_id, tracks_poses, tracks_ts, tracks_cls, tracks_flags, cuboids = (
        [], [], [], [], [], []
    )
    for itoken, data in instances.items():
        tracks_id.append(itoken)
        tracks_poses.append(data["poses"])
        tracks_ts.append(data["timestamps"])
        tracks_cls.append(data["label"])
        tracks_flags.append("DYNAMIC|CONTROLLABLE")
        cuboids.append(data["dim"])

    return {
        sequence_id: {
            "tracks_data": {
                "tracks_id":             tracks_id,
                "tracks_poses":          tracks_poses,
                "tracks_timestamps_us":  tracks_ts,
                "tracks_label_class":    tracks_cls,
                "tracks_flags":          tracks_flags,
            },
            "cuboidtracks_data": {
                "cuboids_dims": cuboids,
            },
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_scene(
    nusc: NuScenes,
    scene_index: int,
    dataroot: str,
    output_dir: str,
    skip_mesh: bool = False,
) -> str:
    """
    Convert scene at index `scene_index` and write a .usdz to `output_dir`.
    Returns the path to the written file.
    """
    scene        = nusc.scene[scene_index]
    artifact_uuid = str(uuid.uuid4())
    sequence_id   = f"clipgt-{scene['token']}"
    today         = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"Scene [{scene_index}]: {scene['name']}")
    print(f"  Description : {scene['description'][:80]}")
    print(f"  sequence_id : {sequence_id}")
    print(f"  artifact_uuid: {artifact_uuid}")
    print(f"{'='*60}")

    # ── rig_trajectories.json ─────────────────────────────────────────
    print("\n[1/4] Building ego trajectory + camera calibrations...")
    rig_json, ego_ts = build_rig_trajectories(nusc, scene, sequence_id)
    print(f"  {len(ego_ts)} keyframe poses  |  "
          f"ts range: [{min(ego_ts)}, {max(ego_ts)}]  |  "
          f"{len(rig_json['camera_calibrations'])} cameras")

    # ── sequence_tracks.json ──────────────────────────────────────────
    print("\n[2/4] Building traffic actor tracks...")
    tracks_json = build_sequence_tracks(nusc, scene, sequence_id)
    n_tracks = len(tracks_json[sequence_id]["tracks_data"]["tracks_id"])
    print(f"  {n_tracks} actors (≥ 2 frames each)")

    # ── mesh ──────────────────────────────────────────────────────────
    mesh_ply_bytes = b""
    if skip_mesh:
        print("\n[3/4] Skipping mesh reconstruction (--skip-mesh).")
    else:
        print("\n[3/4] Building ground mesh from LiDAR...")
        mesh = build_ground_mesh_for_scene(nusc, scene, dataroot)
        mesh_ply_bytes = mesh_to_ply_bytes(mesh)
        print(f"  PLY size: {len(mesh_ply_bytes) / 1024:.1f} KB")

    # ── metadata.yaml ─────────────────────────────────────────────────
    metadata = {
        "scene_id":       sequence_id,
        "uuid":           artifact_uuid,
        "version_string": "nuscenes-v1.0-mini",
        "training_date":  today,
        "dataset_hash":   scene["token"],
        "is_resumable":   False,
        "sensors": {
            "camera_ids": CAM_CHANNELS,
            "lidar_ids":  ["LIDAR_TOP"],
        },
        "logger":     {"name": "nuscenes", "run_id": None, "run_url": None},
        "time_range": {"start": min(ego_ts), "end": max(ego_ts)},
        "training_step_outputs": {},
    }

    # ── assemble USDZ (ZIP archive) ───────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{artifact_uuid}.usdz"

    print(f"\n[4/4] Writing USDZ → {output_path}")
    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.yaml",        yaml.dump(metadata, default_flow_style=False))
        zf.writestr("rig_trajectories.json", json.dumps(rig_json))
        zf.writestr("sequence_tracks.json",  json.dumps(tracks_json))
        # USD stubs (required by NRE; not read by alpasim Python code)
        zf.writestr("default.usda",          MINIMAL_USDA)
        zf.writestr("dome_light.usda",       DOME_LIGHT_USDA)
        zf.writestr("mesh.usd",              MESH_USD_STUB.format(name="Mesh"))
        zf.writestr("mesh_ground.usd",       MESH_USD_STUB.format(name="MeshGround"))
        if mesh_ply_bytes:
            zf.writestr("mesh.ply",          mesh_ply_bytes)
            zf.writestr("mesh_ground.ply",   mesh_ply_bytes)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Written: {size_kb:.1f} KB")

    # ── register in sim_scenes.csv ────────────────────────────────────
    csv_path = Path(dataroot) / "scenes" / "sim_scenes.csv"
    if csv_path.exists():
        rel_path = output_path.relative_to(Path(dataroot))
        new_row = (
            f"{artifact_uuid},{sequence_id},nuscenes-v1.0-mini,"
            f"{rel_path},{today},local\n"
        )
        with open(csv_path, "a") as f:
            f.write(new_row)
        print(f"  Registered in {csv_path}")

    return str(output_path)


def validate_artifact(usdz_path: str) -> None:
    """
    Quick smoke-test: load the USDZ with alpasim's own Artifact class and
    print the parsed fields so you can confirm correctness.
    """
    try:
        import sys
        # Add alpasim utils to path if not already installed
        utils_src = Path(__file__).parent.parent.parent / "utils"
        if str(utils_src) not in sys.path:
            sys.path.insert(0, str(utils_src))

        from alpasim_utils.artifact import Artifact
    except ImportError:
        print("\n[validate] alpasim_utils not importable – skipping validation.")
        return

    print(f"\n{'─'*60}")
    print("Validating with alpasim Artifact class...")
    a = Artifact(source=usdz_path)

    meta = a.metadata
    print(f"  scene_id     : {meta.scene_id}")
    print(f"  uuid         : {meta.uuid}")
    print(f"  time_range   : [{meta.time_range.start}, {meta.time_range.end}]")
    print(f"  cameras      : {meta.sensors.camera_ids}")

    rig = a.rig
    print(f"  rig trajectory: {len(rig.trajectory.timestamps_us)} poses")
    print(f"  vehicle bbox  : {rig.vehicle_config}")

    to = a.traffic_objects
    print(f"  traffic actors: {len(to)}")

    try:
        mesh_bytes = a.mesh_ply
        print(f"  mesh_ground.ply: {len(mesh_bytes) / 1024:.1f} KB")
    except KeyError:
        print("  mesh_ground.ply: not present (skip_mesh was used)")

    print("Validation passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a nuScenes scene to an alpasim-compatible .usdz artifact",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataroot",
        default="/workspace/vla-test/alpasim/data",
        help="nuScenes dataroot (contains v1.0-mini/, samples/, sweeps/, maps/)",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version string",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Index into nusc.scene list (0–9 for mini; run with --list to see all)",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/vla-test/alpasim/data/nre-artifacts/all-usdzs",
        help="Directory to write the .usdz file",
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Skip LiDAR mesh reconstruction (faster; no mesh.ply / mesh_ground.ply)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenes and exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run alpasim Artifact validation after writing",
    )
    args = parser.parse_args()

    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"Loaded {len(nusc.scene)} scenes.")

    if args.list:
        print("\nAvailable scenes:")
        for i, sc in enumerate(nusc.scene):
            print(f"  [{i:2d}] {sc['name']:15s}  {sc['description'][:70]}")
        return

    output_path = convert_scene(
        nusc,
        args.scene_index,
        args.dataroot,
        args.output_dir,
        skip_mesh=args.skip_mesh,
    )

    if args.validate:
        validate_artifact(output_path)

    print(f"\nDone → {output_path}")


if __name__ == "__main__":
    main()
