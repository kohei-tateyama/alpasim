#!/bin/bash
# Run the simulation with manual (keyboard) control.
#
# The manual driver runs inside the driver-0 Docker container (same network
# as the rest of the sim stack), switched to model_type=MANUAL so it never
# loads a checkpoint and uses no GPU memory.
#
# A pygame window will open on your local display ($DISPLAY) showing the
# camera feed.  Use keyboard to drive:
#
#   W / ↑     Accelerate
#   S / ↓     Brake / decelerate
#   A / ←     Steer left
#   D / →     Steer right
#   SPACE     Emergency stop
#   ESC / Q   Quit
set -e

cd "$(dirname "$0")"

# ── Pre-flight: download pygame wheel on the host so the container can
# install it offline (the container has no internet access).
WHEELS_DIR=/tmp/alpasim-wheels
mkdir -p "$WHEELS_DIR"
if ! ls "$WHEELS_DIR"/pygame*.whl &>/dev/null; then
    echo "Downloading pygame wheel (host only, one-time)..."
    pip download pygame --dest "$WHEELS_DIR" -q
fi
export WHEELS_DIR

# Allow the Docker container to open windows on the host X display.
if command -v xhost &>/dev/null && [ -n "${DISPLAY:-}" ]; then
    xhost +local:docker 2>/dev/null || true
fi

echo "Starting simulation in manual mode..."
docker compose \
    -f docker-compose.yaml \
    -f docker-compose.manual.yaml \
    --profile sim up

echo "Done."
