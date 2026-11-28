#!/usr/bin/env bash
set -euo pipefail

REGION= ap-southeast-2
ACCOUNT_ID=433306737796
SECRET_ID=rag-ops
APP_DIR=/opt/Rag-Ops
APP_USER=ubuntu
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

cd "$APP_DIR"

# 4. Write .env from Secrets Manager
apt-get install -y jq
aws secretsmanager get-secret-value \
  --secret-id "$SECRET_ID" \
  --region "$REGION" \
  --query SecretString --output text \
  | jq -r 'to_entries[] | "\(.key)=\(.value)"' > "$APP_DIR/.env"
chmod 600 "$APP_DIR/.env"
chown "$APP_USER":"$APP_USER" "$APP_DIR/.env"
cut -d= -f1 "$APP_DIR/.env"   # sanity check: key names, no values

# 6. Log in to ECR, pull the images, start the stack
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker compose pull
docker compose up -d
docker compose ps