#!/bin/bash

# JUPYTER_TOKEN will empty in development, but set for
# deployments. It will be applied from the environment
# automatically when running `docker-compose up`.
# For more details see `docker-compose.production.yml`,
# `docker-compose.override.yml`, and
# `nginx.conf`.

mkdir -p /workspace/data/earth2cache && cd /workspace/data/earth2cache \
    && wget -q https://dli-lms.s3.us-east-1.amazonaws.com/data/x-fx-82-v1/cache4dli.tar.gz \
    && tar xzf cache4dli.tar.gz --strip-components=1 && rm -f cache4dli.tar.gz
python3 /workspace/exercises/data/fetch_ics.py &

cd /workspace/exercises && jupyter lab \
    --ip 0.0.0.0 \
    --allow-root \
    --no-browser \
    --notebook-dir="/workspace/exercises" \
    --ServerApp.base_url="/lab" \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --ServerApp.password=""
