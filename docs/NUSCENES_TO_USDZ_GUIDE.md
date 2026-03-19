# Creating a USDZ Scene File from nuScenes Data

> **Date:** 2026-03-19  
> **Purpose:** Step-by-step guide to understand the USDZ format used by alpasim and how to produce
> one from nuScenes sensor data.

---

## 1. What Is the USDZ File in alpasim?

Despite the `.usdz` extension (which normally implies an Apple AR/3D format), alpasim's `.usdz`
files are **ZIP archives** that bundle all data required for a single simulation scene. The
extension is a naming convention inherited from the Neural Rendering Engine (NRE) that produced
these files. The core Python class that reads them is
[`Artifact`](../src/utils/alpasim_utils/artifact.py).

The archive is loaded as a `zipfile.ZipFile` and each inner file is read on-demand.

---

## 2. Contents of a USDZ Archive

Inspecting an existing scene
(`data/nre-artifacts/all-usdzs/318cf863-25ea-4d9b-b3b4-e4eb902bf69d.usdz`) reveals:

```
checkpoint.ckpt         ← NRE neural rendering model weights
data_info.json          ← Dataset provenance / split info
datasource_summary.json ← Source dataset statistics
default.usda            ← Top-level USD scene description
dome_light.usda         ← USD lighting setup
map.xodr                ← Road network (OpenDRIVE format)
mesh_ground.ply         ← Ground-only triangle mesh (for physics)
mesh_ground.usd         ← USD wrapper around ground mesh
mesh.ply                ← Full scene triangle mesh
mesh.usd                ← USD wrapper around full mesh
metadata.yaml           ← Scene metadata (IDs, sensors, time range)
parsed_config.yaml      ← NRE training configuration
pose_record.json        ← Raw GNSS/IMU pose log
rig_trajectories.json   ← Ego vehicle trajectory + camera calibrations
rig_trajectories.usda   ← USD animation of ego trajectory
sequence_tracks.json    ← Traffic actor tracks (positions, classes, bboxes)
sequence_tracks.usda    ← USD animation of traffic tracks
volume.nurec            ← Neural rendering volume (NRE proprietary format)
volume.usda             ← USD wrapper referencing volume.nurec
```

### 2.1 Files alpasim reads at runtime

| File | Read by | Purpose |
|---|---|---|
| `metadata.yaml` | `Artifact.metadata` | Scene ID, sensor list, time range, uuid |
| `rig_trajectories.json` | `Artifact.rig` | Ego trajectory, camera calibrations, bbox |
| `sequence_tracks.json` | `Artifact.traffic_objects` | Other actors trajectories + classes |
| `map.xodr` | `Artifact.map` | Road network for metrics / lane association |
| `mesh.ply` or `mesh_ground.ply` | `Artifact.mesh_ply` | Ground mesh for physics service |
| `checkpoint.ckpt` + `volume.nurec` | NRE sensor-sim service | Renders camera images |

---

## 3. Detailed Schema of Each Required File

### 3.1 `metadata.yaml`

```yaml
scene_id: clipgt-<uuid>         # Unique scene identifier prefixed with "clipgt-"
uuid: <hex-uuid>                 # UUID4 of this artifact
version_string: 25.7.9-e633dd23  # NRE version used to produce the artifact
training_date: '2025-07-31'
dataset_hash: <md5>
is_resumable: false
sensors:
  camera_ids:                    # Logical camera names present in rig_trajectories.json
    - camera_front_wide_120fov
    - camera_front_tele_30fov
    - camera_cross_right_120fov
    - camera_cross_left_120fov
    - camera_rear_left_70fov
    - camera_rear_right_70fov
  lidar_ids:
    - lidar_gt_top_p128
logger:
  name: dummy
  run_id: null
  run_url: null
time_range:
  start: <microseconds>          # Start timestamp (Unix time in µs)
  end:   <microseconds>          # End timestamp (Unix time in µs)
training_step_outputs:
  psnr: 32.04                    # Optional: quality metric from neural training
```

### 3.2 `rig_trajectories.json`

Top-level keys:

```json
{
  "T_world_base": [[...4x4 matrix...]],   // ECEF world-to-base transform (for XODR coord align)
  "world_to_nre": {"matrix": [[...4x4...]]},
  "camera_calibrations": { "<cam_id>@<sequence_id>": { ... } },
  "lidar_calibrations": { ... },
  "rig_trajectories": [ { ... } ]
}
```

Each `camera_calibrations` entry:

```json
{
  "sequence_id": "clipgt-...",
  "logical_sensor_name": "camera_front_wide_120fov",
  "unique_sensor_idx": 0,
  "T_sensor_rig": [[...4x4 extrinsic matrix...]],
  "camera_model": { ... }   // intrinsic params (focal length, distortion, etc.)
}
```

Each entry in `rig_trajectories`:

```json
{
  "sequence_id": "clipgt-...",
  "rig_bbox": {
    "centroid": [x_m, y_m, z_m],   // offset from rig origin to bbox center
    "dim":      [length, width, height],
    "rot":      [0.0, 0.0, 0.0]    // must be zero (no rotation)
  },
  "T_rig_worlds": [ [[...4x4...]], ... ],          // pose at each timestamp
  "T_rig_world_timestamps_us": [ 12345678, ... ],  // microsecond timestamps
  "cameras_frame_timestamps_us": { "<cam_id>@<seq_id>": [ ... ] },
  "lidars_frame_timestamps_us":  { "<lidar_id>@<seq_id>": [ ... ] },
  "cameras_linear_start_frame_indices": { ... },
  "lidars_linear_start_frame_indices": { ... }
}
```

`T_rig_worlds` encodes the **pose of the rig (ego vehicle) in world space** as 4×4 homogeneous
matrices. Convention: `T_rig_world` transforms a point in world frame to rig frame (i.e. the
inverse of the ego pose).

### 3.3 `sequence_tracks.json`

A dictionary keyed by `sequence_id`. Each value is a dict:

```json
{
  "tracks_xyz_txyz_qxyzw": [
    [[x, y, z, tx, ty, tz, qx, qy, qz, qw], ...],  // per-frame pose for track 0
    [[...], ...],                                    // track 1
    ...
  ],
  "tracks_timestamps_us": [
    [t0, t1, ...],   // timestamps for track 0
    ...
  ],
  "tracks_label_class": ["automobile", "person", "heavy_truck", ...],
  "tracks_flags": ["NONE", "DYNAMIC|CONTROLLABLE", ...],
  "cuboidtracks_data": {
    "cuboids_dims": [[length, width, height], ...]   // 3D bbox per track
  }
}
```

Pose format for each track frame: `[x, y, z, tx, ty, tz, qx, qy, qz, qw]`  
- `x, y, z` = translation in world space (metres)  
- `qx, qy, qz, qw` = orientation quaternion (xyzw convention)  
- `tx, ty, tz` = apparently redundant (same as xyz in observed data)

Flag meanings:
- `NONE` → static or background actor  
- `DYNAMIC|CONTROLLABLE` → actor that can be driven by trafficsim

### 3.4 `map.xodr`

Standard **OpenDRIVE** XML (`.xodr`) road network file. alpasim parses it with `trajdata` to build
a `VectorMap` for lane-level metric computation. The coordinate system must align with the poses
in `rig_trajectories.json`; this is ensured via `T_world_base` (ECEF → ENU transformation).

### 3.5 `mesh.ply` / `mesh_ground.ply`

A **PLY triangle mesh** representing the 3D reconstruction of the scene. Used by the physics
service to constrain vehicles to the road surface. `mesh_ground.ply` is the ground-only subset;
`mesh.ply` includes surrounding geometry (buildings, vegetation, etc.).

### 3.6 `checkpoint.ckpt` + `volume.nurec`

The **NRE (Neural Rendering Engine)** model weights and neural volume. These are used exclusively
by the sensor-sim service to synthesise camera frames. They are the result of a NeRF / 3DGS
training run on real video data and cannot be hand-crafted — they require running a training
pipeline (e.g. 3DGRUT in this same workspace: `3dgrut/`).

---

## 4. nuScenes Data and What It Provides

[nuScenes](https://www.nuscenes.org/) is a large-scale autonomous driving dataset collected in
Boston and Singapore. A single nuScenes **scene** (20 s clip, 40 keyframes at 2 Hz, ~400k LiDAR
points per sweep) contains:

| nuScenes data | alpasim equivalent |
|---|---|
| Ego pose (`ego_pose`) — translation + quaternion at each timestamp | `T_rig_worlds` in `rig_trajectories.json` |
| Camera calibration (`calibrated_sensor` for 6 cameras) | `camera_calibrations` in `rig_trajectories.json` |
| Camera images (6 × ~1600×900 JPEG at 12 Hz) | Training data for NRE (`checkpoint.ckpt` + `volume.nurec`) |
| LiDAR sweeps (32-beam top LIDAR at 20 Hz) | Source for `mesh.ply` reconstruction |
| 3D bounding box annotations (`sample_annotation`) | `sequence_tracks.json` |
| Map (`nuScenes-map`) — raster + vector | Needs conversion to OpenDRIVE for `map.xodr` |
| GNSS/IMU (`ego_pose` + optional CAN) | `T_world_base` + `pose_record.json` |

---

## 5. Step-by-Step: Building a USDZ from nuScenes

### Step 0 – Prerequisites

```bash
pip install nuscenes-devkit open3d pye57 opendrive2lanelet
# 3DGRUT environment (for neural rendering):
# see /workspace/vla-test/3dgrut/install_env.sh
```

Download the nuScenes mini or full dataset and the nuScenes map expansion.

---

### Step 1 – Choose a Scene and Extract Metadata

```python
from nuscenes.nuscenes import NuScenes
import uuid, yaml

nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes')
scene = nusc.scene[0]                          # pick any scene
sequence_id = f"clipgt-{scene['token']}"       # alpasim scene_id convention
artifact_uuid = str(uuid.uuid4())

metadata = {
    "scene_id": sequence_id,
    "uuid": artifact_uuid,
    "version_string": "nuscenes-v1.0",
    "training_date": "2026-03-19",
    "dataset_hash": scene['token'],
    "is_resumable": False,
    "sensors": {
        "camera_ids": [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
        ],
        "lidar_ids": ["LIDAR_TOP"]
    },
    "logger": {"name": "nuscenes", "run_id": None, "run_url": None},
    "time_range": {"start": 0, "end": 0},   # fill in Step 2
    "training_step_outputs": {}
}
```

---

### Step 2 – Extract Ego Trajectory → `rig_trajectories.json`

nuScenes stores ego pose per sample in world (ENU) coordinates.

```python
import numpy as np
from pyquaternion import Quaternion

def pose_to_T_rig_world(ep):
    """nuScenes ego_pose → 4x4 T_rig_world (world-to-rig)."""
    t = np.array(ep['translation'])
    q = Quaternion(ep['rotation'])
    T_world_rig = np.eye(4)
    T_world_rig[:3, :3] = q.rotation_matrix
    T_world_rig[:3, 3]  = t
    return np.linalg.inv(T_world_rig).tolist()   # T_rig_world

timestamps_us, T_rig_worlds = [], []
sample_token = scene['first_sample_token']
while sample_token:
    sample = nusc.get('sample', sample_token)
    ep = nusc.get('ego_pose', nusc.get('sample_data',
                  sample['data']['LIDAR_TOP'])['ego_pose_token'])
    timestamps_us.append(ep['timestamp'])       # already in µs
    T_rig_worlds.append(pose_to_T_rig_world(ep))
    sample_token = sample['next']

metadata['time_range']['start'] = timestamps_us[0]
metadata['time_range']['end']   = timestamps_us[-1]
```

Build `camera_calibrations` from nuScenes `calibrated_sensor` records:

```python
camera_calibrations = {}
for cam_name in metadata['sensors']['camera_ids']:
    sd = nusc.get('sample_data', nusc.get('sample', scene['first_sample_token'])['data'][cam_name])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    T_sensor_rig = np.eye(4)
    T_sensor_rig[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
    T_sensor_rig[:3, 3]  = cs['translation']
    T_sensor_rig = np.linalg.inv(T_sensor_rig).tolist()  # T_sensor_rig convention

    key = f"{cam_name}@{sequence_id}"
    camera_calibrations[key] = {
        "sequence_id": sequence_id,
        "logical_sensor_name": cam_name,
        "unique_sensor_idx": metadata['sensors']['camera_ids'].index(cam_name),
        "T_sensor_rig": T_sensor_rig,
        "camera_model": {
            "type": "pinhole",
            "intrinsics": cs['camera_intrinsic']    # [[fx,0,cx],[0,fy,cy],[0,0,1]]
        }
    }
```

Assemble `rig_trajectories.json`:

```python
rig_json = {
    "T_world_base": get_ecef_transform(nusc, scene),  # see Step 5
    "world_to_nre": {"matrix": np.eye(4).tolist()},   # identity if not training NRE
    "camera_calibrations": camera_calibrations,
    "lidar_calibrations": {},
    "rig_trajectories": [{
        "sequence_id": sequence_id,
        "rig_bbox": {
            "centroid": [1.44, 0.0, 0.75],   # typical car (adjust to actual vehicle)
            "dim":      [5.29, 2.11, 1.50],
            "rot":      [0.0, 0.0, 0.0]
        },
        "T_rig_worlds": T_rig_worlds,
        "T_rig_world_timestamps_us": timestamps_us,
        "cameras_frame_timestamps_us": build_cam_timestamps(nusc, scene, sequence_id),
        "lidars_frame_timestamps_us": {},
        "cameras_linear_start_frame_indices": {},
        "lidars_linear_start_frame_indices": {}
    }]
}
```

---

### Step 3 – Extract Traffic Tracks → `sequence_tracks.json`

nuScenes annotations (`sample_annotation`) give 3D bboxes + instance tokens per keyframe.

```python
LABEL_MAP = {
    'vehicle.car':         'automobile',
    'vehicle.truck':       'heavy_truck',
    'vehicle.trailer':     'trailer',
    'human.pedestrian.*':  'person',
    'animal':              'animal',
}

tracks = {}   # instance_token → list of per-frame data
for sample_token in iter_samples(scene):
    sample = nusc.get('sample', sample_token)
    ep = nusc.get('ego_pose', ...)
    ts_us = ep['timestamp']
    for ann_token in sample['anns']:
        ann   = nusc.get('sample_annotation', ann_token)
        itoken = ann['instance_token']
        tracks.setdefault(itoken, []).append({
            'timestamp_us': ts_us,
            'xyz':          ann['translation'],      # [x, y, z] in world
            'wlh':          ann['size'],             # [width, length, height]
            'quat':         ann['rotation'],         # [w, x, y, z] → convert to xyzw
            'category':     ann['category_name']
        })

# format to sequence_tracks.json schema
tracks_xyz = []
tracks_ts  = []
tracks_cls = []
tracks_dim = []
for itoken, frames in tracks.items():
    frames_sorted = sorted(frames, key=lambda f: f['timestamp_us'])
    per_frame = []
    for f in frames_sorted:
        x, y, z = f['xyz']
        qw, qx, qy, qz = f['quat']          # nuScenes is [w,x,y,z]
        per_frame.append([x, y, z, x, y, z, qx, qy, qz, qw])
    tracks_xyz.append(per_frame)
    tracks_ts.append([f['timestamp_us'] for f in frames_sorted])
    tracks_cls.append(map_category(frames_sorted[0]['category']))
    w, l, h = frames_sorted[0]['wlh']
    tracks_dim.append([l, w, h])             # alpasim uses [length, width, height]

sequence_tracks = {
    sequence_id: {
        "tracks_xyz_txyz_qxyzw": tracks_xyz,
        "tracks_timestamps_us":  tracks_ts,
        "tracks_label_class":    tracks_cls,
        "tracks_flags":          ["DYNAMIC|CONTROLLABLE"] * len(tracks_cls),
        "cuboidtracks_data":     {"cuboids_dims": tracks_dim}
    }
}
```

---

### Step 4 – Build the Road Map → `map.xodr`

nuScenes ships a **raster + vector map** (not OpenDRIVE). You need to convert it:

**Option A – `nuScenes2HD-map` / `opendrive2lanelet` pipeline:**

```bash
# Export nuScenes map to intermediate format, then convert:
pip install nuscenes-devkit opendrive2lanelet
```

No off-the-shelf converter exists from nuScenes JSON maps to `.xodr`; the practical approaches:

1. **Use `nuscenes-devkit` map API** to export centerlines and lane boundaries as GeoJSON, then
   use a GeoJSON → OpenDRIVE tool such as
   [RoadRunner](https://www.mathworks.com/products/roadrunner.html) or the open-source
   [`road2simulation`](https://github.com/carla-simulator/scenario_runner).
2. **Skip the map** — set `map.xodr` to a minimal placeholder. The simulation will warn but
   continue; map-dependent KPIs (lane-keeping, etc.) will be disabled.
3. **Use a pre-existing OpenDRIVE map** for the same geographic location (e.g. from OpenStreetMap
   via [osm2xodr](https://github.com/stefan-urban/osm2xodr)) aligned to the coordinate frame.

---

### Step 5 – Build the Ground Mesh → `mesh.ply` / `mesh_ground.ply`

Aggregate LiDAR sweeps into the world frame and reconstruct a surface mesh:

```python
import open3d as o3d, numpy as np

all_pts = []
for each_lidar_sweep:
    pts_sensor = load_points(sweep)                  # (N, 3)
    T_world_sensor = get_T_world_sensor(sweep)
    pts_world = (T_world_sensor[:3,:3] @ pts_sensor.T).T + T_world_sensor[:3, 3]
    all_pts.append(pts_world)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
pcd.estimate_normals()

# Poisson surface reconstruction
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
mesh = mesh.simplify_quadric_decimation(500_000)

# Ground-only: filter by z-height and normal orientation
# (keep faces where normal ≈ vertical and z < threshold)
ground = filter_ground(mesh)

o3d.io.write_triangle_mesh("mesh.ply",        mesh)
o3d.io.write_triangle_mesh("mesh_ground.ply", ground)
```

---

### Step 6 – Neural Rendering Volume → `checkpoint.ckpt` + `volume.nurec`

This is the **hardest step** and requires significant compute (GPU hours).

The 3DGRUT project (also in this workspace at `3dgrut/`) is the intended training pipeline:

```bash
# 1. Prepare data in NeRF-compatible format from nuScenes images + poses:
python prepare_nuscenes_for_3dgrut.py --scene-token <token> --output /data/scene/

# 2. Train the 3DGS/3DGRUT model:
cd /workspace/vla-test/3dgrut
python train.py \
    --config configs/base_gs.yaml \
    experiment.name=my_nuscenes_scene \
    data.path=/data/scene/

# 3. Export to NRE format:
python render.py --config ... --export-usdz /output/volume.nurec
```

The output `checkpoint.ckpt` (model weights) and `volume.nurec` (NRE volume) are placed into the
archive. Without this step the sensor-sim service cannot render camera images, but the simulation
can still run in **GT-only mode** (using ground-truth images from the dataset directly).

---

### Step 7 – Assemble the USDZ Archive

```python
import zipfile, json, yaml

def write_usdz(output_path, files: dict):
    """files: {archive_name: bytes_or_str}"""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode()
            zf.writestr(name, content)

write_usdz(f"{artifact_uuid}.usdz", {
    "metadata.yaml":          yaml.dump(metadata),
    "rig_trajectories.json":  json.dumps(rig_json),
    "sequence_tracks.json":   json.dumps(sequence_tracks),
    "map.xodr":               open("map.xodr").read(),
    "mesh.ply":               open("mesh.ply","rb").read(),
    "mesh_ground.ply":        open("mesh_ground.ply","rb").read(),
    "checkpoint.ckpt":        open("checkpoint.ckpt","rb").read(),  # from Step 6
    "volume.nurec":           open("volume.nurec","rb").read(),     # from Step 6
    # USD wrappers (minimal stubs are fine if NRE is not used):
    "default.usda":           MINIMAL_USDA_STUB,
})
```

---

### Step 8 – Register the Scene in `sim_scenes.csv`

Add a row to [data/scenes/sim_scenes.csv](../data/scenes/sim_scenes.csv):

```csv
uuid,scene_id,nre_version_string,path,last_modified,artifact_repository
<artifact_uuid>,clipgt-<scene_token>,nuscenes-v1.0,<relative_path>/<uuid>.usdz,2026-03-19,local
```

Then point `--usdz-glob` in your run config to the location of your new file.

---

## 6. Data Requirements Summary

| What you need | Source in nuScenes | Notes |
|---|---|---|
| Ego trajectory (poses) | `ego_pose` table | One pose per LiDAR sweep |
| Camera intrinsics & extrinsics | `calibrated_sensor` table | Per camera, static per scene |
| Camera timestamps | `sample_data` table | 6 cameras × 12 Hz |
| Vehicle dimensions | Annotation or known spec | nuScenes ego = Ford Fusion |
| 3D actor tracks + classes | `sample_annotation` table | Keyframes only (2 Hz), interpolation needed |
| LiDAR sweeps | `sample_data` (LIDAR_TOP) | For mesh reconstruction |
| Road map | nuScenes map API or OpenStreetMap | Needs conversion to OpenDRIVE |
| ECEF coordinate transform | GNSS metadata | Required for map → sim alignment |
| Neural rendering | Camera images + poses | GPU training ~4–24 h per scene |

---

## 7. Coordinate Conventions

alpasim uses a **right-handed coordinate system**:
- `+X` → forward  
- `+Y` → left  
- `+Z` → up

nuScenes uses the same handedness but `+X` forward, `+Y` left, `+Z` up in the ego frame, so no
axis-flip is needed for poses. However, the global (world) frame in nuScenes is arbitrary ENU;
you need `T_world_base` (an ECEF 4×4 matrix) so the XODR map loader can align the map to the
trajectory. This can be obtained from nuScenes scene `location` + the ego GPS coordinates.

Quaternion convention: nuScenes uses `[w, x, y, z]`; alpasim's `sequence_tracks.json` uses
`[qx, qy, qz, qw]` (xyzw). **Always swap** when writing track data.

---

## 8. What Can Be Skipped for a Minimal Simulation

If you only want to validate the ego trajectory and actor behaviour (without real camera renders):

- **Skip** `checkpoint.ckpt` + `volume.nurec` → use a dummy NRE or replay mode  
- **Skip** `map.xodr` → map-based KPIs disabled, simulation still runs  
- **Use a flat plane** for `mesh.ply` / `mesh_ground.ply` → physics service runs with degraded
  terrain accuracy

The minimum viable USDZ only needs: `metadata.yaml`, `rig_trajectories.json`,
`sequence_tracks.json`, and `mesh_ground.ply`.

---

## 9. Key Files in This Codebase

| File | Role |
|---|---|
| [src/utils/alpasim_utils/artifact.py](../src/utils/alpasim_utils/artifact.py) | Reads and parses every field of the USDZ |
| [src/utils/alpasim_utils/scenario.py](../src/utils/alpasim_utils/scenario.py) | Defines `Rig`, `TrafficObjects`, `VehicleConfig` data classes |
| [src/runtime/alpasim_runtime/simulate/__main__.py](../src/runtime/alpasim_runtime/simulate/__main__.py) | Consumes USDZ via `--usdz-glob` argument |
| [data/scenes/sim_scenes.csv](../data/scenes/sim_scenes.csv) | Registry of all known scene artifacts |
| [data/nre-artifacts/all-usdzs/](../data/nre-artifacts/all-usdzs/) | Example USDZ files to inspect |
| [3dgrut/train.py](../../3dgrut/train.py) | Neural rendering training (produces checkpoint + volume) |
