#!/usr/bin/env bash
# 本地 → 服务器 同步脚本（rsync over ssh，需要本机已装 sshpass 或配置 ssh key）
# 用法：bash scripts/sync_to_server.sh
set -euo pipefail

LOCAL_DIR="/Users/zouzeyu/Documents/Code/0512/"
REMOTE_USER="admin"
REMOTE_HOST="192.168.7.36"
REMOTE_DIR="/home/admin/multimodel_search/0512/"
PASSWORD="root@NAS1024"

EXCLUDES=(
  --exclude ".git"
  --exclude "__pycache__"
  --exclude ".venv"
  --exclude ".env"
  --exclude "*.pyc"
  --exclude ".DS_Store"
  --exclude "logs/"
  --exclude "data/"
)

if command -v sshpass >/dev/null 2>&1; then
  sshpass -p "${PASSWORD}" rsync -avz --delete "${EXCLUDES[@]}" \
    -e "ssh -o StrictHostKeyChecking=no" \
    "${LOCAL_DIR}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
else
  rsync -avz --delete "${EXCLUDES[@]}" \
    -e "ssh -o StrictHostKeyChecking=no" \
    "${LOCAL_DIR}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
fi

echo "[sync] done → ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
