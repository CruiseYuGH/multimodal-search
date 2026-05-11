#!/usr/bin/env bash
# 启动 Milvus standalone（docker），数据持久化到 /home/admin/multimodel_search/0512/milvus_data
set -euo pipefail

DATA_DIR="/home/admin/multimodel_search/0512/milvus_data"
CONTAINER_NAME="milvus-standalone-0512"
IMAGE="milvusdb/milvus:v2.4.4"
HOST_PORT=19530

# 检查是否已运行
if docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[milvus] 容器 ${CONTAINER_NAME} 已在运行"
  exit 0
fi

# 检查是否已存在但停止
if docker ps -a --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[milvus] 容器 ${CONTAINER_NAME} 已存在，启动中..."
  docker start "${CONTAINER_NAME}"
  exit 0
fi

# 创建数据目录
mkdir -p "${DATA_DIR}/etcd" "${DATA_DIR}/minio" "${DATA_DIR}/milvus"

# 首次创建容器
echo "[milvus] 创建并启动容器 ${CONTAINER_NAME}..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart=unless-stopped \
  -p ${HOST_PORT}:19530 \
  -p 9091:9091 \
  -e ETCD_USE_EMBED=true \
  -e COMMON_STORAGETYPE=local \
  -v "${DATA_DIR}/milvus:/var/lib/milvus" \
  -v "${DATA_DIR}/etcd:/etcd" \
  -v "${DATA_DIR}/minio:/minio" \
  "${IMAGE}" \
  milvus run standalone

echo "[milvus] 等待 Milvus 就绪（最多 60s）..."
for i in {1..60}; do
  if docker exec "${CONTAINER_NAME}" curl -s http://localhost:9091/healthz | grep -q "OK"; then
    echo "[milvus] 就绪 ✓"
    exit 0
  fi
  sleep 1
done

echo "[milvus] 启动超时，请检查日志：docker logs ${CONTAINER_NAME}"
exit 1
