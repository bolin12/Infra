# 开发日志：Python 环境初始化

**日期**: 2026-07-06  
**机器**: RTX 4060 / CachyOS  
**目的**: 用 uv 安装 Python 3.12 + LLM 推理加速工具链

---

## 一、系统环境

| 项目 | 详情 |
|------|------|
| OS | CachyOS (Arch-based) |
| Kernel | Linux 7.1.3-1-cachyos |
| GPU | NVIDIA GeForce RTX 4060 (8 GB) |
| Driver | 610.43.02 |
| CUDA | 13.3 (system), 13.0 (PyTorch 内置) |
| 系统 Python | 3.14.6 (未使用) |

---

## 二、安装步骤

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# → uv 0.11.26, installed to ~/.local/bin
```

### 2. 安装 Python 3.12.13

```bash
uv python install 3.12
# → cpython-3.12.13-linux-x86_64-gnu

# 项目初始化
uv init --python 3.12 --no-readme
```

### 3. 配置 pyproject.toml

核心依赖分组：
- **Core ML**: torch, transformers, accelerate, safetensors, tokenizers, datasets, huggingface-hub
- **Inference Engines**: vllm, sglang, llama-cpp-python
- **加速**: triton, xformers, bitsandbytes, flashinfer (vllm 自动拉入)
- **量化**: optimum
- **工具**: numpy, pandas, matplotlib, seaborn, einops, sentencepiece, protobuf, psutil

PyTorch 索引: `https://download.pytorch.org/whl/cu124` → 实际拉到了 cu130 (2.11.0)

### 4. 安装命令（最终版：一条命令）

```bash
uv sync --python 3.12 --index-strategy unsafe-best-match
# → 218 packages installed，包括 llama-cpp-python（自动编译）和 cmake

# 之前需要手动安装 cmake + --no-build-isolation 的 workaround 已不需要
# 通过 [tool.uv.extra-build-dependencies] 声明 cmake 为 build 依赖即可
```

---

## 三、已安装包清单

### 推理引擎

| 包 | 版本 | 用途 |
|----|------|------|
| vllm | 0.20.2 | 高性能 LLM serving |
| sglang | 0.5.2 | 结构化生成 + serving |
| llama-cpp-python | 0.3.33 | llama.cpp Python 绑定 (GGUF) |

### 加速库

| 包 | 版本 | 用途 |
|----|------|------|
| triton | 3.6.0 | GPU kernel 编写与优化 |
| xformers | 0.0.35 | 高效 attention (C++ 扩展未加载，Python API 可用) |
| bitsandbytes | 0.49.2 | 量化 (8-bit/4-bit) |
| flashinfer | 0.6.8.post1 | 高性能 attention/norm/sampling kernels |

### 模型框架

| 包 | 版本 |
|----|------|
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| transformers | 5.13.0 |
| accelerate | 1.14.0 |
| optimum | 2.2.0 |
| tokenizers | 0.22.2 |
| datasets | 5.0.0 |
| huggingface-hub | 1.22.0 |
| safetensors | 0.8.0 |
| tiktoken | 0.13.0 |

### 数据与分析

| 包 | 版本 |
|----|------|
| numpy | 2.3.5 |
| pandas | 3.0.3 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| pyarrow | 24.0.0 |

### CUDA 库 (nvidia-*)

| 包 | 版本 |
|----|------|
| nvidia-cublas | 13.1.0.3 |
| nvidia-cudnn-cu13 | 9.19.0.56 |
| nvidia-nccl-cu13 | 2.28.9 |
| nvidia-cusparselt-cu13 | 0.8.0 |
| nvidia-cutlass-dsl | 4.6.0 |

---

## 四、跳过的包 & 原因

| 包 | 原因 |
|----|------|
| auto-gptq | QiGen CUDA kernel 生成失败，CUDA 13.3 不兼容 |
| autoawq | 未尝试（同样需 CUDA 编译，预计类似问题） |
| flash-attn | 未尝试（CUDA 13.3 兼容性问题） |
| exllamav2 | 未尝试（需 GitHub 源码编译） |

> **替代方案**: GGUF 量化用 llama-cpp-python 已可用；GPTQ/AWQ 量化对比实验待这些包更新后再做。

---

## 五、环境激活方式

```bash
# 激活 venv
source /home/lbl/Codes/Infra/.venv/bin/activate.fish

# 或直接用 uv 运行
uv run python script.py

# 查看已安装
uv pip list
```

---

## 六、验证结果

```
PyTorch: 2.11.0+cu130 | GPU: RTX 4060 (7.6 GB)
transformers 5.13.0, vllm 0.20.2, sglang 0.5.2
triton 3.6.0, flashinfer 0.6.8, llama.cpp 0.3.33
GPU MatMul 4096×4096: ✓
```

---

## 七、下一步

1. 运行 `bash scripts/collect_env.sh` 保存完整硬件信息
2. Nsight Systems / Nsight Compute 已本地安装，profile 时用
3. 下载测试模型 (2B/4B/8B)
4. 开始 `experiments/01_rtx4060_baseline`
