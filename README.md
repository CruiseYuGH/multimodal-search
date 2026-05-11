# multimodal-search (0512)

本地多模态搜索：text / image / text-to-image / hybrid 查询，Milvus 存储，Haystack 2.x 编排。

- 文本空间：BGE-M3 (dim=1024, IP, 内置 L2)
- 图文空间：fg-clip2 (dim=768, IP, 出口手动 L2)
- 视频：保留接口位，未实现

## 目录
- app/        业务代码
- config/     全局配置
- scripts/    初始化 / 同步 / 冒烟
- tests/      单测占位

## 工作流
1. 本地改代码 → 2. scripts/sync_to_server.sh 同步到服务器 → 3. 服务器跑 scripts/smoke_test.py → 4. 本地 git push
