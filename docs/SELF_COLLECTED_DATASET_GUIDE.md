# Collecting Your Own Dataset to Run alpasim

> **Purpose:** Complete reference of every sensor, calibration measurement, and recording
> procedure you need to collect data from your own vehicle and convert it into a USDZ scene file
> that alpasim can simulate in.

---

## 1. Hardware You Need to Install on the Car

### 1.1 Cameras

alpasim's NRE (Neural Rendering Engine) synthesises new camera views at runtime.
The quality and coverage of the neural reconstruction depends entirely on the cameras used during
recording.

| Property | Requirement | Notes |
|---|---|---|
| **Number** | Minimum 1 (front), ideally 4–6 | More cameras → better 360° reconstruction |
| **Placement** | Front (wide + tele), sides, rear | Match the logical names you'll use in `metadata.yaml` |
| **Resolution** | ≥ 1280×720 per camera | Higher is better for neural training; 1920×1080 recommended |
| **Frame rate** | ≥ 10 Hz, ideally 12–30 Hz | Must be hardware-synchronised or have accurate per-frame timestamps |
| **Shutter** | Global or rolling (note which) | Must be recorded in `CameraDefinitionConfig.shutter_type` |
| **Exposure** | Short / auto-HDR | Motion blur degrades neural reconstruction |
| **Lens type** | Pinhole, fisheye, or f-theta | alpasim supports all three via `CameraIntrinsicsConfig` |
| **Format** | JPEG or RAW | JPEG acceptable for NRE training; RAW preserves more dynamic range |

**Suggested layout** (matches the existing NRE camera naming convention):

```
camera_front_wide_120fov    — wide front (≈ 120° horizontal FOV)
camera_front_tele_30fov     — telephoto front (≈ 30° horizontal FOV)
camera_cross_left_120fov    — left-side, pointing 90° left
camera_cross_right_120fov   — right-side, pointing 90° right
camera_rear_left_70fov      — rear-left diagonal
camera_rear_right_70fov     — rear-right diagonal
```

You can use fewer cameras; just list only the ones you have in `metadata.yaml`.

---

### 1.2 LiDAR

Used to reconstruct the 3D ground mesh (`mesh.ply`, `mesh_ground.ply`) and to detect/track other
traffic participants.

| Property | Requirement |
|---|---|
| **Type** | Spinning or solid-state (spinning preferred for 360°) |
| **Beams / lines** | ≥ 32; 64 or 128 recommended for good mesh quality |
| **Range** | ≥ 80 m |
| **Rate** | ≥ 10 Hz |
| **Point format** | XYZ + intensity + timestamp per point |
| **Mounting** | Roof centre, unobstructed 360° view |

---

### 1.3 GNSS + IMU (Pose System)

This is the most critical sensor for alpasim. The ego pose
(`T_rig_worlds` in `rig_trajectories.json`) must be accurate.

| Property | Requirement |
|---|---|
| **GNSS accuracy** | RTK-GNSS preferred (cm-level); single-antenna GNSS acceptable (1–3 m) |
| **IMU** | 6-DoF (accel + gyro), ≥ 100 Hz |
| **Fusion** | Use a GNSS-INS (inertial navigation) fusion system (e.g. Applanix, NovAtel, OxTS) |
| **Output** | Position (lat/lon/alt or ECEF XYZ), orientation (roll/pitch/yaw or quaternion), timestamps |
| **ECEF transform** | Required for aligning the road map to the trajectory (see §3.6) |
| **Timestamp sync** | GNSS PPS signal used to hardware-sync all other sensors |

> **Minimum viable option:** A phone GPS + 6-DoF IMU gives ~3–5 m accuracy and is enough to
> validate the pipeline, but neural rendering quality will suffer from pose noise.

---

### 1.4 Vehicle CAN / OBD

Optional but useful for recording ground-truth vehicle dynamics.

| Signal | Use in alpasim |
|---|---|
| Speed (m/s) | Validates ego trajectory, used by physics service |
| Steering angle | Vehicle model / controller comparison |
| Wheel speeds | Dead-reckoning fallback for GPS gaps |
| Vehicle dimensions | Stored in `rig_bbox` (`length`, `width`, `height`, centroid offset) |

---

### 1.5 Time Synchronisation

**All sensors must share a common clock.** This is non-negotiable.

- Use a **GPS PPS signal** (1 pulse per second) as the hardware trigger for cameras and LiDAR.
- The IMU typically runs on the same clock as the GNSS receiver.
- Store all timestamps as **Unix time in microseconds (µs)** — this is what alpasim uses
  everywhere (`timestamps_us`).

---

## 2. What to Measure / Calibrate Before Driving

### 2.1 Camera Intrinsics (per camera)

Calibrate each camera individually using a checkerboard / charuco board.
Record all of the following:

| Parameter | Symbol | alpasim field |
|---|---|---|
| Focal length (x, y) | `fx`, `fy` | `focal_length: [fx, fy]` |
| Principal point | `cx`, `cy` | `principal_point: [cx, cy]` |
| Radial distortion | `k1..k6` | `radial: [k1, k2, k3, k4, k5, k6]` |
| Tangential distortion | `p1`, `p2` | `tangential: [p1, p2]` |
| Thin prism distortion | `s1..s4` | `thin_prism: [s1, s2, s3, s4]` (optional) |
| Image resolution | H × W | `resolution_hw: [H, W]` |
| Shutter type | — | `shutter_type: "ROLLING_TOP_TO_BOTTOM"` etc. |

For **fisheye** lenses use the OpenCV fisheye model (`k1..k4`, `max_angle`).  
For **f-theta** lenses use the polynomial `pixeldist_to_angle` or `angle_to_pixeldist`.  
See [`config.py`](../src/runtime/alpasim_runtime/config.py) for all three model definitions.

A helper script exists to convert OpenCV pinhole calibrations to f-theta:
[`src/tools/scripts/pinhole_to_ftheta.py`](../src/tools/scripts/pinhole_to_ftheta.py)

---

### 2.2 Camera-to-IMU Extrinsics (per camera)

The pose of each camera **relative to the vehicle rig origin** (= IMU / rear-axle centre).
This goes into `T_sensor_rig` in `rig_trajectories.json`.

| Parameter | How to measure |
|---|---|
| Translation (x, y, z) in metres | Tape-measure from IMU origin to camera optical centre |
| Rotation (quaternion xyzw) | Surveying equipment, or perform sensor-fusion online calibration (e.g. `kalibr`) |

The rig origin convention used by alpasim (inherited from NVIDIA NDAS) is:
**"on the ground directly below the centre of the rear axle"**.

```
+X = forward
+Y = left
+Z = up
```

---

### 2.3 LiDAR-to-IMU Extrinsics

Same as camera — translation + rotation of the LiDAR optical centre relative to IMU origin.
Used when transforming point clouds into the world frame for mesh reconstruction.

---

### 2.4 Vehicle Bounding Box (for `rig_bbox`)

Measure your car's geometry. alpasim uses this for collision detection and physics.

| Field | Description | How to measure |
|---|---|---|
| `dim.length` | Full vehicle length (m) | Tape measure front bumper to rear bumper |
| `dim.width` | Full vehicle width (m) | Tape measure mirror to mirror |
| `dim.height` | Full vehicle height (m) | Ground to roof |
| `centroid.x` | Distance forward from rear-axle centre to bbox centre | `(length/2) - rear_overhang` |
| `centroid.y` | Lateral offset (usually 0) | 0 for symmetric vehicles |
| `centroid.z` | Height from ground to bbox centre | `height / 2` |
| `rot` | Rotation of bbox relative to rig | Always `[0, 0, 0]` |

Example entry in `rig_trajectories.json`:

```json
"rig_bbox": {
    "centroid": [1.45, 0.0, 0.75],
    "dim":      [4.60, 1.85, 1.50],
    "rot":      [0.0, 0.0, 0.0]
}
```

---

## 3. Data to Record While Driving

### 3.1 Ego Pose (→ `rig_trajectories.json`)

Record the 6-DoF pose of the IMU/rig at **≥ 10 Hz**. Each pose becomes one row in
`T_rig_worlds` (a 4×4 homogeneous matrix, world-to-rig convention).

| Field | Rate | Format |
|---|---|---|
| Position | ≥ 10 Hz | ENU (East-North-Up) in metres, or ECEF XYZ |
| Orientation | ≥ 10 Hz | Quaternion (xyzw) or Euler angles (roll, pitch, yaw) |
| Timestamp | Every pose | Unix microseconds (`uint64`) |

Converting ENU pose to `T_rig_world` (4×4, world-to-rig):

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def enu_to_T_rig_world(translation_enu, quat_xyzw):
    T_world_rig = np.eye(4)
    T_world_rig[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T_world_rig[:3, 3]  = translation_enu
    return np.linalg.inv(T_world_rig).tolist()  # T_rig_world
```

---

### 3.2 Camera Frames (→ NRE training input)

For each camera, record:

| Field | Requirement |
|---|---|
| Image file (JPEG/PNG) | Every frame |
| Frame timestamp (µs) | Each image — must match the ego pose timeline |
| Camera ID | Logical name (e.g. `camera_front_wide_120fov`) |
| Sequence ID | Same as scene `scene_id` |

These images are the training data for 3DGRUT / NRE, which produces `checkpoint.ckpt` and
`volume.nurec`. Without good synchronised images you cannot train a usable neural rendering model.

**Minimum recording length:** 20–60 seconds at slow speed (≤ 30 km/h) with the vehicle driving
in a loop or straight-and-back to give the neural model multi-view coverage of the scene.

---

### 3.3 LiDAR Sweeps (→ `mesh.ply`, `mesh_ground.ply`)

For each sweep (typically at 10–20 Hz), record:

| Field | Requirement |
|---|---|
| Point cloud (XYZ + intensity) | Per sweep |
| Per-point timestamp (µs) | Needed for motion de-skew during aggregation |
| Sweep timestamp | Unix µs at scan midpoint |
| Sensor pose at sweep time | From GNSS-INS log (same timestamps) |

All sweeps are aggregated in the world frame and then surface-reconstructed:

```python
# Pseudocode
for sweep in lidar_sweeps:
    T_world_lidar = get_pose_at(sweep.timestamp_us) @ T_lidar_rig
    pts_world = (T_world_lidar[:3,:3] @ sweep.points.T).T + T_world_lidar[:3, 3]
    all_pts.append(pts_world)
# → Poisson surface reconstruction → mesh.ply
```

---

### 3.4 Traffic Actor Annotations (→ `sequence_tracks.json`)

You need the 3D bounding boxes and labels of every other vehicle, pedestrian, cyclist etc. in the
scene over time. This data goes into `sequence_tracks.json` and drives the traffic simulation.

alpasim expects this schema:

```json
{
  "<sequence_id>": {
    "tracks_data": {
      "tracks_id":             ["<unique_id_per_actor>", ...],
      "tracks_label_class":    ["automobile", "person", "heavy_truck", ...],
      "tracks_flags":          ["DYNAMIC|CONTROLLABLE", "NONE", ...],
      "tracks_timestamps_us":  [[t0, t1, ...], [t0, t1, ...], ...],
      "tracks_poses":          [[[x,y,z, qx,qy,qz,qw], ...], ...]
    },
    "cuboidtracks_data": {
      "cuboids_dims": [[length, width, height], ...]
    }
  }
}
```

**Pose format per track frame:** `[x, y, z, qx, qy, qz, qw]`  
- `x, y, z` — world position (metres, same frame as ego poses)  
- `qx, qy, qz, qw` — orientation quaternion (xyzw / scipy convention)

**Label classes** (use exactly these strings):

| Class string | Vehicle type |
|---|---|
| `automobile` | Passenger car, SUV |
| `heavy_truck` | Semi-truck, lorry |
| `trailer` | Trailer attached to a truck |
| `person` | Pedestrian |
| `animal` | Animal on road |
| `motorcycle` | Motorbike, scooter |
| `bicycle` | Bicycle |

**Flags:**

| Flag | Meaning |
|---|---|
| `DYNAMIC|CONTROLLABLE` | Actor can be driven by the traffic simulator |
| `NONE` | Static obstacle / parked vehicle |

**How to get annotations:**

1. **LiDAR-based 3D detection:** Run an off-the-shelf detector such as
   [CenterPoint](https://github.com/tianweiy/CenterPoint) or
   [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) on your LiDAR sweeps.
2. **Multi-camera detection + LiDAR fusion:** Use a model like DETR3D or BEVFusion.
3. **Manual annotation:** Use tools like [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS)
   or [3D Bounding Box Annotation Tool](https://github.com/walzimmer/3d-bat).
4. **Tracking:** Pass detections through a tracker (e.g. SimpleTrack, SORT) to get consistent
   `track_id` values across frames.

> **Important:** Timestamps in `tracks_timestamps_us` must be in **the same clock** as the ego
> pose timestamps. Use the LiDAR sweep timestamp as the per-detection timestamp.

---

### 3.5 Road Map (→ `map.xodr`)

alpasim needs an OpenDRIVE (`.xodr`) road network covering the area of your recording.

**Option A — OSM export + conversion (free):**

```bash
# 1. Export the road area from OpenStreetMap (use the bounding box of your recording)
#    at https://www.openstreetmap.org/export
# 2. Convert to OpenDRIVE:
pip install osm2xodr
python -m osm2xodr input.osm --output map.xodr
```

**Option B — HD map provider:** Use commercial sources (HERE HD Live Map, TomTom, Mobileye Road
Experience Management) if you need lane-level accuracy.

**Option C — Record-and-generate:** Tools like
[cruse](https://github.com/MaikRo/cruse) can generate a rough XODR from a recorded trajectory.

The map coordinate frame must match your ego trajectory coordinate frame. Use `T_world_base`
(an ECEF 4×4 matrix computed from your GNSS latitude/longitude) to perform the alignment — see
[`artifact.py`](../src/utils/alpasim_utils/artifact.py) `_get_xodr_transform()`.

---

### 3.6 ECEF Coordinate Transform (→ `T_world_base` in `rig_trajectories.json`)

This 4×4 matrix aligns the simulation's local world frame with the geodetic ECEF frame so that
`map.xodr` and the ego trajectory share the same coordinate system.

Compute it from the GNSS origin of your recording:

```python
import numpy as np
from pyproj import Transformer

def ecef_from_lla(lat_deg, lon_deg, alt_m):
    tf = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    return tf.transform(lon_deg, lat_deg, alt_m)

def compute_T_world_base(lat0, lon0, alt0):
    """
    Returns a 4x4 matrix: ECEF → local ENU frame centred at (lat0, lon0, alt0).
    This is the value to store in T_world_base.
    """
    x0, y0, z0 = ecef_from_lla(lat0, lon0, alt0)

    sin_lat, cos_lat = np.sin(np.radians(lat0)), np.cos(np.radians(lat0))
    sin_lon, cos_lon = np.sin(np.radians(lon0)), np.cos(np.radians(lon0))

    # Rotation: ECEF → ENU
    R = np.array([
        [-sin_lon,             cos_lon,           0        ],
        [-sin_lat * cos_lon,  -sin_lat * sin_lon,  cos_lat  ],
        [ cos_lat * cos_lon,   cos_lat * sin_lon,  sin_lat  ],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = -R @ np.array([x0, y0, z0])
    return T.tolist()
```

Store the first ego pose's GPS location as the origin `(lat0, lon0, alt0)`.

---

## 4. Recording Checklist

Use this before and during each data collection drive.

### Before the Drive

- [ ] Camera intrinsics calibrated for each lens
- [ ] Camera-to-IMU extrinsics calibrated (`T_sensor_rig`)
- [ ] LiDAR-to-IMU extrinsics calibrated
- [ ] Vehicle bounding box measured (`length`, `width`, `height`, centroid offsets)
- [ ] All sensor clocks synchronised to GNSS PPS
- [ ] Storage available: ≥ 10 GB per minute of recording (6 cameras + LiDAR + GNSS)
- [ ] GNSS RTK base station active (if using RTK)
- [ ] Recording software started and logging confirmed

### During the Drive

- [ ] Speed ≤ 30 km/h for best neural reconstruction (sharp imagery, dense point cloud)
- [ ] Cover the scene with multiple passes for multi-view neural training
- [ ] Avoid heavy motion blur (short exposure times)
- [ ] Record ≥ 20 s (ideally 30–60 s) of continuous driving

### After the Drive

- [ ] Verify GNSS trajectory continuity (no large jumps)
- [ ] Verify camera frame counts match expected timestamps
- [ ] Verify LiDAR sweep count matches expected rate
- [ ] Check that all clock offsets are within ±1 ms

---

## 5. Data Volume Estimates

| Sensor | Rate | Size per minute |
|---|---|---|
| 6× 1080p cameras @ 12 Hz | 72 frames/s total | ≈ 1.5–3 GB |
| LiDAR (128-beam, 20 Hz) | 2560 points/beam | ≈ 3–5 GB |
| GNSS-INS poses @ 100 Hz | — | ≈ 5 MB |
| CAN / OBD @ 100 Hz | — | ≈ 2 MB |
| **Total per minute** | | **≈ 5–8 GB** |

For a 60-second scene: plan for **8–10 GB raw data** and **50–100 GB working storage** during
processing (mesh reconstruction, neural training).

---

## 6. Processing Pipeline After Collection

```
Raw data
    │
    ├── GNSS-INS poses ────────────────────────────── rig_trajectories.json
    │       │
    │       └── T_world_base (from GPS origin) ───── rig_trajectories.json
    │
    ├── Camera images + poses + calibrations ──────── NRE training (3DGRUT)
    │                                                      │
    │                                                      ├── checkpoint.ckpt
    │                                                      └── volume.nurec
    │
    ├── LiDAR sweeps + poses ──────────────────────── mesh.ply + mesh_ground.ply
    │
    ├── 3D detection + tracking on LiDAR sweeps ───── sequence_tracks.json
    │
    ├── OSM / HD map ──────────────────────────────── map.xodr
    │
    └── Measured vehicle geometry + sensor metadata ─ metadata.yaml
                                                           │
                                                           └── All files → ZIP → scene.usdz
```

---

## 7. Key Files in the Codebase (Reference)

| File | Reads / defines |
|---|---|
| [`src/utils/alpasim_utils/artifact.py`](../src/utils/alpasim_utils/artifact.py) | Parses all USDZ fields at runtime |
| [`src/utils/alpasim_utils/scenario.py`](../src/utils/alpasim_utils/scenario.py) | `Rig`, `TrafficObjects`, `VehicleConfig`, `QVec` pose format |
| [`src/runtime/alpasim_runtime/config.py`](../src/runtime/alpasim_runtime/config.py) | Camera models: `OpenCVPinholeConfig`, `OpenCVFisheyeConfig`, `FthetaConfig` |
| [`src/tools/scripts/pinhole_to_ftheta.py`](../src/tools/scripts/pinhole_to_ftheta.py) | Converts OpenCV pinhole calibration to f-theta polynomial |
| [`docs/NUSCENES_TO_USDZ_GUIDE.md`](NUSCENES_TO_USDZ_GUIDE.md) | Full USDZ assembly walkthrough (nuScenes as worked example) |

---

## 8. Minimum Viable Dataset (No Neural Rendering)

If you want to run simulations **without** training a neural rendering model (no NRE camera
synthesis), you only need:

| File | What to record/provide |
|---|---|
| `metadata.yaml` | Scene ID, sensor names, time range |
| `rig_trajectories.json` | GNSS-INS ego poses + camera calibrations |
| `sequence_tracks.json` | 3D actor detections + tracking |
| `mesh_ground.ply` | LiDAR ground plane reconstruction |

In this mode the sensor-sim service is bypassed and the simulation runs **replay-mode only**
(the driver receives no camera images). This is useful for:
- Testing the physics service
- Validating actor tracks
- Running the evaluation/metrics pipeline on a pre-recorded trajectory
