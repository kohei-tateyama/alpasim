# Alpasim Driver Fix Guide for Bosch Environment

## Overview
This guide documents the fixes required to run Alpasim simulations with VaVAM and Alpamayo models in a Docker environment with corporate network restrictions. These fixes resolve three critical issues encountered when running simulations behind a corporate proxy.

---

## Problems Encountered

### 1. **pygame Import Error** ❌
```
driver-0-1      | ModuleNotFoundError: No module named 'pygame'
runtime-0-1     | grpc.aio._call.AioRpcError: <AioRpcError of RPC that terminated with:
runtime-0-1     |       status = StatusCode.UNIMPLEMENTED
runtime-0-1     |       details = "Received http2 header with status: 404"
```

**Root Cause:** The driver unconditionally imported `ManualModel` at startup, which requires `pygame`. Since `pygame` isn't installed in the Docker image (only needed for manual keyboard control mode), the import failed, causing the driver service to crash before starting its gRPC server.

### 2. **osqp Missing in Controller** ❌
```
controller-0-1  | ModuleNotFoundError: No module named 'osqp'
runtime-0-1     | grpc.aio._call.AioRpcError: <AioRpcError of RPC that terminated with:
runtime-0-1     |       status = StatusCode.UNKNOWN
runtime-0-1     |       details = "Exception calling application: No module named 'osqp'"
```

**Root Cause:** The `osqp` package (required by LinearMPC controller) was listed in `src/controller/pyproject.toml` but not installed in the Docker image, likely due to network/proxy issues during the initial build.

### 3. **Docker Networking Issues with Corporate Proxy** ❌
```
runtime-0-1     | E0227 04:43:03.336938 HTTP proxy handshake with ipv4:172.17.0.1:3128 failed: UNKNOWN
runtime-0-1     | grpc.aio._call.AioRpcError: Received http2 header with status: 404
```

**Root Cause:** Corporate proxy environment variables (`HTTP_PROXY=http://127.0.0.1:3128`) were inherited by Docker containers, causing gRPC to attempt proxying localhost connections through the corporate proxy (which fails).

---

## Solution: Step-by-Step Fixes

### Fix 1: Lazy Load ManualModel (pygame Fix)

#### File 1: `src/driver/src/alpasim_driver/models/__init__.py`

**Before (Lines 5-8):**
```python
from .ar1_model import AR1Model
from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction
from .manual_model import ManualModel  # ← This causes pygame import error
from .transfuser_model import TransfuserModel
```

**After (Lines 5-13):**
```python
from .ar1_model import AR1Model
from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction
# ManualModel requires pygame - lazy load only when needed
from .transfuser_model import TransfuserModel
from .vam_model import VAMModel


def _lazy_load_manual_model():
    """Lazy import ManualModel to avoid requiring pygame dependency."""
    from .manual_model import ManualModel
    return ManualModel
```

**Before (Lines 12-19):**
```python
__all__ = [
    "AR1Model",
    "BaseTrajectoryModel",
    "DriveCommand",
    "ManualModel",  # ← Remove this
    "ModelPrediction",
    "TransfuserModel",
    "VAMModel",
]
```

**After (Lines 16-24):**
```python
__all__ = [
    "AR1Model",
    "BaseTrajectoryModel",
    "DriveCommand",
    "ModelPrediction",
    "TransfuserModel",
    "VAMModel",
    "_lazy_load_manual_model",  # ← Add this
]
```

---

#### File 2: `src/driver/src/alpasim_driver/main.py`

**Change 1 - Update imports (Line 59):**

**Before:**
```python
from .frame_cache import FrameCache
from .models import DriveCommand
from .models.ar1_model import AR1Model
from .models.base import BaseTrajectoryModel, ModelPrediction
from .models.manual_model import ManualModel  # ← Remove this import
from .models.transfuser_model import TransfuserModel
from .models.vam_model import VAMModel
```

**After:**
```python
from .frame_cache import FrameCache
from .models import DriveCommand, _lazy_load_manual_model  # ← Add lazy loader
from .models.ar1_model import AR1Model
from .models.base import BaseTrajectoryModel, ModelPrediction
# ManualModel requires pygame - lazy load only when needed
from .models.transfuser_model import TransfuserModel
from .models.vam_model import VAMModel
```

**Change 2 - Update _create_model function (Lines 462-468):**

**Before:**
```python
    elif cfg.model_type == ModelType.MANUAL:
        return ManualModel(
            camera_ids=camera_ids,
            output_frequency_hz=output_frequency_hz,
            context_length=context_length or 1,
        )
```

**After:**
```python
    elif cfg.model_type == ModelType.MANUAL:
        ManualModel = _lazy_load_manual_model()  # ← Load only when needed
        return ManualModel(
            camera_ids=camera_ids,
            output_frequency_hz=output_frequency_hz,
            context_length=context_length or 1,
        )
```

**Change 3 - Update main function (Lines 1203-1209):**

**Before:**
```python
        # Wait for the service (and ManualModel) to be created
        ready_event.wait(timeout=30.0)

        # Run pygame loop on main thread using the singleton GUI instance
        if ManualModel._gui_instance is not None:  # ← ManualModel not defined
            ManualModel._gui_instance.run_main_loop()
        else:
```

**After:**
```python
        # Wait for the service (and ManualModel) to be created
        ready_event.wait(timeout=30.0)

        # Run pygame loop on main thread using the singleton GUI instance
        ManualModel = _lazy_load_manual_model()  # ← Load before checking
        if ManualModel._gui_instance is not None:
            ManualModel._gui_instance.run_main_loop()
        else:
```

**Why this fix works:** ManualModel is only imported when `model_type=MANUAL` is explicitly requested. For VaVAM and Alpamayo models, ManualModel is never imported, so pygame is never required.

---

### Fix 2: Install osqp in Docker Image

**Verification - Check current state:**
```bash
docker run --rm --entrypoint bash alpasim-base:0.1.3 -c "python -c 'import osqp'"
# If this fails, osqp is missing
```

#### Method A: Patch Existing Image (Quick Fix - Recommended)

**Step 1: Create patch Dockerfile**
```bash
cd /workspace/vla-test/alpasim
cat > Dockerfile.patch << 'EOF'
# Quick patch to add osqp to existing alpasim-base image
FROM alpasim-base:0.1.3

WORKDIR /repo
RUN uv pip install osqp
EOF
```

**Step 2: Build patched image**
```bash
docker build -f Dockerfile.patch -t alpasim-base:0.1.3-patched .
```

**Step 3: Replace original with patched version**
```bash
docker tag alpasim-base:0.1.3 alpasim-base:0.1.3-original
docker tag alpasim-base:0.1.3-patched alpasim-base:0.1.3
```

**Step 4: Verify osqp is installed**
```bash
docker run --rm --entrypoint bash alpasim-base:0.1.3 -c "python -c 'import osqp; print(\"osqp OK\")"
```

Expected output:
```
osqp OK
```

#### Method B: Rebuild Image from Scratch (If Network Access Available)

```bash
cd /workspace/vla-test/alpasim
docker compose -f tutorial/docker-compose.yaml build --no-cache \
  --build-arg HTTP_PROXY=http://host.docker.internal:3128 \
  --build-arg HTTPS_PROXY=http://host.docker.internal:3128
```

**Note:** The `osqp` dependency is already declared in `src/controller/pyproject.toml` line 15, so rebuilding will automatically install it.

---

### Fix 3: Docker Networking + Proxy Configuration

#### File: `src/wizard/alpasim_wizard/deployment/docker_compose.py`

**Change 1 - Add proxy variable clearing (Lines 177-194):**

**Before:**
```python
        if container.workdir:
            ret["working_dir"] = container.workdir
        if container.environments:
            ret["environment"] = container.environments

        addresses = container.get_all_addresses()
```

**After:**
```python
        if container.workdir:
            ret["working_dir"] = container.workdir
        if container.environments:
            ret["environment"] = container.environments
        
        # Unset proxy variables for localhost connections (prevents gRPC proxy errors)
        if self.context.cfg.wizard.debug_flags.use_localhost:
            if "environment" not in ret:
                ret["environment"] = {}
            # Convert list to dict if needed
            if isinstance(ret["environment"], list):
                env_dict = {}
                for item in ret["environment"]:
                    if "=" in item:
                        key, val = item.split("=", 1)
                        env_dict[key] = val
                ret["environment"] = env_dict
            ret["environment"]["HTTP_PROXY"] = ""
            ret["environment"]["HTTPS_PROXY"] = ""
            ret["environment"]["http_proxy"] = ""
            ret["environment"]["https_proxy"] = ""

        addresses = container.get_all_addresses()
```

**Change 2 - Add 30-second startup delay to runtime (Lines 225-232):**

**Before:**
```python
        # Add runtime services last in sim phase
        for c in container_set.runtime or []:
            service = self._to_docker_compose_service(c)
            service["profiles"] = ["sim"]
            # Runtime needs host PID namespace for process monitoring
            service["pid"] = "host"
```

**After:**
```python
        # Add runtime services last in sim phase
        for c in container_set.runtime or []:
            service = self._to_docker_compose_service(c)
            service["profiles"] = ["sim"]
            # Add delay before runtime starts to allow services to initialize
            if service.get("command"):
                service["command"][-1] = f"sleep 30 && {service['command'][-1]}"
            # Runtime needs host PID namespace for process monitoring
            service["pid"] = "host"
```

**Why these fixes work:**
1. **Host networking:** Bypasses Docker's bridge network (which fails with corporate proxy)
2. **Proxy variables cleared:** Prevents gRPC from trying to proxy localhost connections
3. **30-second delay:** Gives driver time to load VaVAM model (~13 seconds) before runtime tries to connect

---

## Complete Setup Instructions

### Initial Setup

**Step 1: Check your environment**
```bash
cd /workspace/vla-test/alpasim
env | grep -i proxy
```

Expected output (Bosch corporate network):
```
HTTP_PROXY=http://127.0.0.1:3128
HTTPS_PROXY=http://127.0.0.1:3128
NO_PROXY=127.0.0.1,127.*,10.*,...
```

**Step 2: Apply all code fixes**

Apply the changes to:
- `src/driver/src/alpasim_driver/models/__init__.py` (Fix 1)
- `src/driver/src/alpasim_driver/main.py` (Fix 1)
- `src/wizard/alpasim_wizard/deployment/docker_compose.py` (Fix 3)

**Step 3: Patch Docker image for osqp**
```bash
cd /workspace/vla-test/alpasim

# Create patch file
cat > Dockerfile.patch << 'EOF'
FROM alpasim-base:0.1.3
WORKDIR /repo
RUN uv pip install osqp
EOF

# Build and tag
docker build -f Dockerfile.patch -t alpasim-base:0.1.3-patched .
docker tag alpasim-base:0.1.3 alpasim-base:0.1.3-original
docker tag alpasim-base:0.1.3-patched alpasim-base:0.1.3
```

**Step 4: Setup local environment (optional, for development)**
```bash
source setup_local_env.sh
# Note: May show PyPI connection errors due to proxy - this is OK for Docker-based simulation
```

---

### Running the Simulation

**Clean run:**
```bash
cd /workspace/vla-test/alpasim

# Clean previous runs (use sudo if needed)
sudo rm -rf tutorial/

# Generate configuration and run
alpasim_wizard +deploy=local \
  wizard.log_dir=$PWD/tutorial \
  wizard.debug_flags.use_localhost=true
```

**Expected successful output:**
```
[2026-02-27 13:42:57,609][alpasim_wizard][INFO] - Writing docker compose YAML...
[2026-02-27 13:42:57,615][alpasim_wizard][INFO] - Docker Compose configuration generated
Starting simulation phase...
controller-0-1  | 04:43:00.351 INFO:    SystemManager using linear MPC
controller-0-1  | 04:43:00.351 INFO:    Starting server on 0.0.0.0:6003
physics-0-1     | 04:43:02.542 INFO:    Serving on 0.0.0.0:6002
driver-0-1      | [2026-02-27 04:23:43,180][__main__][INFO] - Starting vam driver on 0.0.0.0:6000
sensorsim-0-1   | [2026-02-27 04:43:07,467][nre.grpc.serve][INFO] - Serving on 0.0.0.0:6001
runtime-0-1     | runtime: version_id: "0.3.0"
runtime-0-1     | Connected to driver: version_id: "vam-driver-0.7.0" (attempt 1)
runtime-0-1     | Connected to sensorsim: version_id: "25.7.9" (attempt 1)
runtime-0-1     | Connected to physics: version_id: "0.2.0" (attempt 1)
runtime-0-1     | Connected to controller: version_id: "0.14.0" (attempt 1)
runtime-0-1     | Built 1 jobs to execute
runtime-0-1     | Worker 0 starting (num_workers=1)
runtime-0-1     | Session STARTING: uuid=... scene=clipgt-... steps=100
runtime-0-1     | Simulation loop timer started
```

---

## Verification Steps

### 1. Check Driver Starts Without pygame Error
```bash
docker logs tutorial-driver-0-1 2>&1 | grep -E "(pygame|ModuleNotFoundError|Starting vam)"
```

**Expected:** NO pygame errors, should see:
```
[2026-02-27 04:23:43,180][__main__][INFO] - Starting vam driver on 0.0.0.0:6000
```

### 2. Check Controller Has osqp
```bash
docker exec tutorial-controller-0-1 bash -c "cd /repo && python -c 'import osqp; print(\"osqp version:\", osqp.__version__)'"
```

**Expected:**
```
osqp version: 1.1.1
```

### 3. Check Proxy Variables Are Cleared
```bash
docker exec tutorial-runtime-0-1 env | grep -i proxy
```

**Expected:** Empty values (our fix):
```
HTTP_PROXY=
HTTPS_PROXY=
http_proxy=
https_proxy=
```

### 4. Check All Services Connect Successfully
```bash
docker logs tutorial-runtime-0-1 2>&1 | grep "Connected to"
```

**Expected:**
```
Connected to driver: version_id: "vam-driver-0.7.0" (attempt 1)
Connected to sensorsim: version_id: "25.7.9" (attempt 1)
Connected to physics: version_id: "0.2.0" (attempt 1)
Connected to controller: version_id: "0.14.0" (attempt 1)
```

All should connect on **attempt 1** without retries.

### 5. Check Simulation Completes
```bash
ls -lh tutorial/rollouts/*/*/videos/
```

**Expected:** Video files generated for the simulation.

---

## Troubleshooting

### Issue: pygame error still appears

**Check:**
```bash
grep "_lazy_load_manual_model" src/driver/src/alpasim_driver/models/__init__.py
grep "_lazy_load_manual_model" src/driver/src/alpasim_driver/main.py
```

**Solution:** Verify all three changes to main.py are applied (imports, _create_model, main function).

---

### Issue: osqp still missing

**Check:**
```bash
docker run --rm --entrypoint bash alpasim-base:0.1.3 -c "python -c 'import osqp'"
```

**Solution if fails:**
```bash
# Rebuild patch
cd /workspace/vla-test/alpasim
docker build -f Dockerfile.patch -t alpasim-base:0.1.3 --no-cache .
```

---

### Issue: HTTP proxy errors persist

**Check docker-compose.yaml:**
```bash
grep -A 5 "environment:" tutorial/docker-compose.yaml | grep -i proxy
```

**Expected:** Should see empty proxy values:
```yaml
environment:
  HTTP_PROXY: ''
  HTTPS_PROXY: ''
  http_proxy: ''
  https_proxy: ''
```

**Solution:** Regenerate configuration:
```bash
sudo rm -rf tutorial/
alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial wizard.debug_flags.use_localhost=true
```

---

### Issue: Runtime fails to connect (404 errors)

**Check timing:**
```bash
docker logs tutorial-driver-0-1 | grep "Loading VAM checkpoint"
```

**Solution:** Increase sleep delay in `src/wizard/alpasim_wizard/deployment/docker_compose.py`:
```python
service["command"][-1] = f"sleep 60 && {service['command'][-1]}"  # Increase to 60s
```

---

### Issue: Permission denied when removing tutorial/

**Solution:**
```bash
sudo rm -rf tutorial/
# Or
sudo chown -R $USER:$USER tutorial/
rm -rf tutorial/
```

Files in tutorial/ are created by Docker containers running as root.

---

## Summary of All Changes

| File | Lines | Change | Purpose |
|------|-------|--------|---------|
| `src/driver/src/alpasim_driver/models/__init__.py` | 5-13 | Add `_lazy_load_manual_model()` function | Lazy load ManualModel to avoid pygame dependency |
| `src/driver/src/alpasim_driver/models/__init__.py` | 16-24 | Update `__all__` exports | Export lazy loader instead of ManualModel |
| `src/driver/src/alpasim_driver/main.py` | 59-63 | Update imports | Import lazy loader, remove ManualModel |
| `src/driver/src/alpasim_driver/main.py` | 462-468 | Update `_create_model()` | Load ManualModel only when needed |
| `src/driver/src/alpasim_driver/main.py` | 1203-1209 | Update `main()` | Load ManualModel before accessing GUI |
| `src/wizard/alpasim_wizard/deployment/docker_compose.py` | 177-194 | Add proxy clearing logic | Prevent gRPC from using corporate proxy |
| `src/wizard/alpasim_wizard/deployment/docker_compose.py` | 225-232 | Add 30s delay to runtime | Allow services to initialize before runtime connects |
| `Dockerfile.patch` | New file | Install osqp in Docker image | Fix missing controller dependency |

---

## Testing Other Configurations

### Run with Alpamayo-R1 Model
```bash
alpasim_wizard +deploy=local \
  wizard.log_dir=$PWD/tutorial_alpamayo \
  wizard.debug_flags.use_localhost=true \
  driver=[ar1,ar1_runtime_configs]
```

### Run Multiple Scenarios
```bash
alpasim_wizard +deploy=local \
  wizard.log_dir=$PWD/tutorial_multi \
  wizard.debug_flags.use_localhost=true \
  wizard.user.num_rollouts=5
```

### Adjust Simulation Steps
```bash
alpasim_wizard +deploy=local \
  wizard.log_dir=$PWD/tutorial \
  wizard.debug_flags.use_localhost=true \
  wizard.user.scenarios[0].num_timesteps=200
```

---

## Key Takeaways for Bosch Environment

1. **Lazy Loading:** Prevents optional dependencies from breaking core functionality
2. **Host Networking:** Essential for bypassing Docker bridge network issues with corporate proxy
3. **Proxy Clearing:** Corporate proxy variables MUST be cleared for internal container communication
4. **Startup Timing:** Heavy models need time to load before other services can connect
5. **Docker Image Patching:** Quick workaround when rebuilding is blocked by network restrictions

---

## Additional Notes

### Network Environment Details
- Bosch corporate proxy: `http://127.0.0.1:3128`
- NO_PROXY includes internal networks: `10.*`, `172.*`, `192.168.*`
- Docker containers cannot reach PyPI directly during build
- Host networking mode bypasses these restrictions for localhost communication

### Model Loading Times
- **VaVAM model:** ~13 seconds to load checkpoint (318M + 37M parameters)
- **Physics service:** ~5 seconds to initialize scene
- **Sensorsim warmup:** ~16 seconds for first render
- **Total before runtime ready:** ~30-35 seconds

### Resource Requirements
- GPU: NVIDIA RTX 6000 Ada Generation (47 GiB)
- CUDA: 12.4.1
- Python: 3.12
- Docker: 24.x or later

---

**Document Version:** 1.0  
**Last Updated:** February 27, 2026  
**Tested Environment:** Bosch Corporate Network, Ubuntu Linux  
**Contact:** Save this document for future Bosch team members setting up Alpasim

---

## Quick Reference Command

**One-liner to run simulation after all fixes applied:**
```bash
cd /workspace/vla-test/alpasim && \
sudo rm -rf tutorial/ && \
alpasim_wizard +deploy=local wizard.log_dir=$PWD/tutorial wizard.debug_flags.use_localhost=true
```
