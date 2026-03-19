#!/bin/bash
set -e

# Run simulation profile
echo "Starting simulation phase..."
docker compose -f docker-compose.yaml --profile sim up

echo "All phases completed successfully!"
