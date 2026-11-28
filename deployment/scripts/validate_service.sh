#!/bin/bash
for i in $(seq 1 30); do
  curl -fs http://localhost:8000/health && exit 0
  sleep 5
done
echo "API did not become healthy" >&2
exit 1