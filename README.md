# 多模态搜索系统

基于 BGE-M3（文本）和 fg-clip2（图文）的本地多模态检索系统，支持文档、图片索引与混合检索。

## 特性

- **多模态索引**：txt / docx / xlsx / pptx / image
- **多种检索模式**：
  - 文本检索（BGE-M3 → 文档库）
  - 图像检索（fg-clip2 image → 图片库）
  - 文本→图像检索（fg-clip2 text → 图片库）
  - 混合检索（多路并行 + RRF/加权融合）
- **向量存储**：Milvus standalone（HNSW 索引）
- **精确定位**：文档返回页码/段落，PPT 返回 slide，Excel 返回 sheet

## 系统要求

- Python 3.9+
- CUDA 11.8+ / ROCm（推荐 GPU，CPU 可用但慢）
- Docker（用于 Milvus）
- 磁盘空间：模型权重 ~10GB，Milvus 数据按索引量增长

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/CruiseYuGH/multimodal-search.git
cd multimodal-search
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型权重

```bash
# BGE-M3
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3

# fg-clip2
huggingface-cli download TencentARC/FlagVision-CLIP2 --local-dir models/fg-clip2
```

### 5. 配置

编辑 `config/settings.py`，修改以下路径：

```python
BGE_M3_MODEL_PATH = "/path/to/models/bge-m3"
FGCLIP2_MODEL_PATH = "/path/to/models/fg-clip2"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
DEVICE = "cuda"  # 或 "cpu"
```

### 6. 启动 Milvus

```bash
bash scripts/start_milvus.sh
```

等待 30 秒后初始化 collections：

```bash
python scripts/init_milvus.py
```

## 使用

### CLI 命令

#### 管理命令

```bash
# 检查 Milvus 连接
python -m app.api.cli admin ping

# 列出支持的文件类型
python -m app.api.cli admin list-loaders

# 查看向量库统计
python -m app.api.cli admin stats
```

#### 索引文件

```bash
# 索引单个文件
python -m app.api.cli index --file /path/to/document.pdf

# 强制重新索引（覆盖已存在）
python -m app.api.cli index --file /path/to/image.png --no-skip-exists
```

#### 检索

**文本检索**（查文档）：

```bash
python -m app.api.cli search text --q "机器学习算法" --topk 10
```

**图像检索**（以图搜图）：

```bash
python -m app.api.cli search image --image /path/to/query.jpg --topk 5
```

**文本→图像检索**（用文字描述搜图）：

```bash
python -m app.api.cli search t2i --q "一只橘猫在阳光下" --topk 5
```

**混合检索**（文本 + 图像多路融合）：

```bash
# RRF 融合（默认）
python -m app.api.cli search hybrid \
  --q "现代简约风格" \
  --image /path/to/reference.jpg \
  --topk 10 \
  --fusion rrf

# 加权融合
python -m app.api.cli search hybrid \
  --q "现代简约风格" \
  --topk 10 \
  --fusion weighted \
  --weight-doc 1.0 \
  --weight-t2i 2.0 \
  --weight-image 1.5
```

### Python API

```python
from app.pipelines.index_pipeline import index_file
from app.pipelines.query_text import query_text
from app.pipelines.query_hybrid import query_hybrid

# 索引
result = index_file("/path/to/file.pdf")
print(f"插入 {result['inserted']} 条向量")

# 文本检索
hits = query_text("机器学习", topk=5)
for h in hits:
    print(f"{h.source_path} (page {h.page}): {h.score:.4f}")

# 混合检索
hits = query_hybrid(
    q="现代简约",
    image_path="/path/to/ref.jpg",
    topk=10,
    fusion="rrf"
)
```

## 测试

```bash
# B 阶段：预处理
python scripts/smoke_preprocess.py

# C 阶段：向量化
python scripts/smoke_embedders.py

# D 阶段：Milvus 存取
python scripts/smoke_store.py

# E 阶段：端到端 pipelines
python scripts/smoke_pipelines.py

# F 阶段：CLI 完整流程
python scripts/smoke_test.py
```

## 架构

```
输入文件
  ↓
preprocess（文本分块 / 图像预处理）
  ↓
embedders（BGE-M3 / fg-clip2）
  ↓
Milvus（mm_doc / mm_image collections）
  ↓
query pipelines（单路 / 混合）
  ↓
ranking（RRF / 加权融合）
  ↓
结果（路径 + 页码 + 分数）
```

### 目录结构

```
├── app/
│   ├── core/           # 数据类型、异常、注册表
│   ├── preprocess/     # 文件加载器（txt/docx/xlsx/pptx/image）
│   ├── embedders/      # 向量化模型（BGE-M3/fg-clip2）
│   ├── store/          # Milvus 客户端 + DocStore/ImageStore
│   ├── pipelines/      # 索引 + 查询流程
│   ├── ranking/        # 分数融合（RRF/加权）
│   └── api/            # CLI 入口
├── config/             # 全局配置
├── scripts/            # 初始化 + 测试脚本
└── tests/              # 单元测试（预留）
```

## 配置说明

`config/settings.py` 关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BGE_M3_DIM` | 1024 | BGE-M3 向量维度 |
| `FG_CLIP2_DIM` | 768 | fg-clip2 向量维度 |
| `CHUNK_SIZE` | 800 | 文本分块字符数 |
| `CHUNK_OVERLAP` | 100 | 分块重叠字符数 |
| `HNSW_M` | 16 | HNSW 图连接数 |
| `HNSW_EF_CONSTRUCTION` | 200 | 索引构建参数 |
| `HNSW_EF_SEARCH` | 64 | 搜索参数 |
| `DEFAULT_TOPK` | 10 | 默认返回结果数 |
| `RRF_K` | 60 | RRF 融合参数 |

## 性能优化

- **fp16**：默认开启，显存减半，精度无损
- **批量索引**：单文件调用，外部脚本可并行
- **HNSW 参数**：M=16/efC=200 适合百万级数据，更大规模可调高
- **GPU 选择**：`DEVICE="cuda:0"` 指定 GPU 卡号

## 已知限制

- **视频支持**：接口已预留，暂未实现（抛 `VideoReserved`）
- **PDF 支持**：需额外依赖（pymupdf/pdfplumber），暂未集成
- **分布式**：当前单机 Milvus standalone，生产环境建议 cluster 模式
- **增量更新**：按 file_hash 去重，修改文件需手动删除旧向量后重新索引

## 故障排查

**Milvus 连接失败**：

```bash
docker ps | grep milvus  # 检查容器状态
docker logs milvus-standalone-0512  # 查看日志
```

**模型加载 OOM**：

- 降低 batch size（修改 embedders 内 `batch_size` 参数）
- 关闭 fp16（`use_fp16=False`）
- 使用 CPU（`DEVICE="cpu"`）

**检索无结果**：

- 确认文件已索引：`python -m app.api.cli admin stats`
- 检查向量归一化：运行 `scripts/smoke_embedders.py`

## 许可证

MIT

## 贡献

欢迎提 Issue 和 PR。

## 联系

- GitHub: https://github.com/CruiseYuGH/multimodal-search
- Email: zouzeyu@example.com
