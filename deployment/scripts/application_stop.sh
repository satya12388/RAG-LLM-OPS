#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/Rag-Ops
[ -f "$APP_DIR/docker-compose.yml" ] || exit 0

cd "$APP_DIR"
docker compose down --rmi all