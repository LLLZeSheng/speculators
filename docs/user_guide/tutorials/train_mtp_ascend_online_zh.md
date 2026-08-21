# Ascend 910C 多机在线训练 GLM-5.2 MTP3

本文支持 4V4T（64-rank FSDP）、4V2T（32-rank FSDP）以及 2V6T
（96-rank FSDP）。每台机器使用 16 张 NPU，verifier 采用 DP1 × TP16。
原有 4V4T 配置和行为保持不变。

数据生命周期如下：

```text
epoch 1：在线请求 verifier → 生成 hidden states → 写入共享缓存 → 训练
epoch 2+：读取同一份缓存 → 按离线方式继续训练
```

第一轮采用 `--on-missing generate --on-generate cache`。后续访问已完成的
样本时不会再次请求 verifier。严格离线恢复采用 `--on-missing raise`，缓存
缺失时立即失败，避免混入不完整数据。

## 1. 固定环境

```text
镜像：quay.io/ascend/vllm-ascend:v0.23.0rc1-a3
共享盘：/mnt/xds/mtp
代码：/mnt/xds/mtp/spec_train/speculators
量化 verifier：
  /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
原始 verifier 权重：
  /mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1
原生 MTP 初始化模型：
  /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
训练数据：/mnt/xds/mtp/spec_train/dataset/hf/nuoya-average2k8k-32k
```

MG13 主模型层仍然是只用于推理的 W4A8 权重，但准备好的 v4 运行视图中，
第 78 层原生 MTP 以及共享 embedding/lm_head 是浮点权重。因此该目录现在
也是首选的 `MTP_INIT_MODEL_PATH`。Speculators 会从 `mtp.safetensors`（或
ModelSlim 索引）读取 MTP，并在转换前拒绝任何整数量化的可训练权重；共享
权重通过 `quant_model_weights.safetensors.index.json` 定位。整个过程不会
修改原模型或运行视图。

所有节点应当具备：

- 相同的 `/mnt/xds/mtp` 挂载；
- 相同的代码路径；
- 可互通的训练网络；
- 控制节点到所有配置机器的免密 SSH；
- Docker 和 16 张可用 Ascend NPU。

## 2. 自己定义并维护 YAML

YAML 是你维护的唯一配置源。管理脚本只读取和校验，不会生成、覆盖或回写
YAML。先复制模板，然后直接填写机器、容器和路径信息：

```bash
cd /mnt/xds/mtp/spec_train/speculators
mkdir -p /mnt/xds/mtp/spec_train/config
cp examples/train/mtp_glm52_ascend_online_4v4t.example.yaml \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
vim /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
```

如果使用 4 verifier + 2 trainer，改用独立模板：

```bash
cp examples/train/mtp_glm52_ascend_online_4v2t.example.yaml \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v2t.yaml
```

4V2T 不需要其他开关。manager 自动设置 `NNODES=2`，trainer0 的本地 rank
在 verifier0/2 之间轮询，trainer1 的本地 rank 在 verifier1/3 之间轮询，
因此四台 verifier 都会参与在线 hidden-state 生成。4V4T 仍按 trainer `i`
对应 verifier `i`。

如果使用 2 verifier + 6 trainer 的 4K MTP2 拓扑：

```bash
cp examples/train/mtp_glm52_ascend_online_2v6t_4k.example.yaml \
  /mnt/xds/mtp/spec_train/config/glm52-mtp2-2v6t-4k.yaml
```

manager 会设置 `NNODES=6` 和 `NODE_RANK=0..5`。由于 verifier 少于
trainer，每台 trainer 都会获得两个 verifier endpoint，再由共享的在途
token lease 在公共池中做全局负载均衡。

需要自行填写或确认：

- `verifier_ips` 中四台 verifier 的 IP；
- `trainer_ips` 中两台、四台或六台 trainer 的 IP；
- `container_mode`、容器名称、镜像和挂载列表；
- 代码、模型、数据、hidden states、checkpoint 和日志路径；
- 所有 `FILL_*` 占位值必须被替换；
- `create` 模式的容器名会自动成为 `glm52-online-mtp3-verifier0`、
  `glm52-online-mtp3-trainer0` 等；`existing` 模式复用
  `existing_container_name`。

建议配置文件放在：

```text
/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
```

启动任何远端任务前先进行纯本地校验：

```bash
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
CONFIG=/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
bash "$MANAGER" validate-config --config "$CONFIG"
```

校验会拒绝重复 IP、未替换占位符、非法容器名前缀、错误的 DP×TP、长度预算
不匹配等问题。`nic_name: auto` 会根据每台机器的 IP 自动识别网卡。
密码不会写入配置；非默认 SSH 设置通过环境变量传入：

```bash
export SSH_USER=root
export SSH_IDENTITY_FILE=/root/.ssh/id_ed25519
```

如需只查看远端命令而不建立 SSH 连接或启动容器：

```bash
MANAGER_DRY_RUN=1 bash \
  examples/train/manage_mtp_glm52_ascend_online_4v4t.sh start-verifiers
```

手写 YAML 建议按 32K 样本设置以下值：

```yaml
total_seq_len: 32768
verifier_max_model_len: 32769
verifier_max_batched_tokens: 32784
verifier_max_num_seqs: 1
request_timeout: 900
max_retries: 3
```

`32769` 为一次 hidden-state 请求额外保留 1 个生成 token。TP16 sequence
parallel 会把 profile token 数向上补齐到 16 的倍数，因此 batched-token 预算
必须设为 `32784`。如果也设成 `32769`，补齐后的 profile 会超过内部 buffer
上限并触发断言。
每台 verifier 是 DP1 × TP16，满 32K 时实际运行 1 条；其余请求会排队。
TP16 为 61 GiB NPU 留出足够的模型分片和 32K profile 激活空间。
`MAX_NUM_SEQS=1`、eager 模式、关闭 shared-expert 多流重叠以及
`PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64` 用于给 61 GiB NPU 留出 32K
prefill 激活空间。`expandable_segments` 不能与 `max_split_size_mb` 同时启用。
900 秒超时用于覆盖长 prefill、排队和 hidden-state 写盘时间。

YAML 同时明确记录容器行为：

```yaml
container_image: quay.io/ascend/vllm-ascend:v0.23.0rc1-a3
container_mode: create
container_name_prefix: glm52-w4a8-mg13-speculator-training
container_repo_path: /mnt/xds/mtp/spec_train/speculators
repo_path: /mnt/xds/mtp/spec_train/speculators
container_mounts:
  - /mnt/xds/sfs:/mnt/xds/sfs
  - /mnt/xds/mtp:/mnt/xds/mtp
  - /mnt/xds/mtp/spec_train/speculators:/mnt/xds/mtp/spec_train/speculators
  - /root/.cache:/root/.cache
install_speculators_verifier: false
install_speculators_trainer: false
```

`container_mode: create` 不需要、也不应填写 `existing_container_name`。
它使用 `docker run` 创建容器，参数与标准 A3 启动方式对齐：host 网络、
默认 1 GiB shm、16 张 NPU、管理设备、Ascend 驱动、
`/mnt/xds/sfs`、`/mnt/xds/mtp` 和 `/root/.cache`。需要额外挂载目录时，直接在
`container_mounts` 追加 `宿主机路径:容器路径[:ro|rw]`。管理任务通过
nohup 后台执行，因此不会加入只能在交互终端使用的 `-it`。

如果所有配置机器上已经提前创建好同名容器，使用：

```yaml
container_mode: existing
existing_container_name: glm52-w4a8-mg13-speculator-training
```

此时管理器通过 `docker exec` 运行任务，`container_image`、
`container_shm_size` 和 `container_mounts` 不会重新应用；这些必须在原容器
创建时已经设置正确。`stop` 只根据 PID 文件停止本次 MTP 任务，不会执行
`docker stop`，因此不会关闭或删除已有容器。

如果已有容器正是用本文开头那种只挂载 `/mnt/xds/mtp:/mnt/xds/mtp` 的命令创建，
容器内并不存在 `/workspace/speculators` 映射。这时应把 YAML 改为实际可见
的代码目录，例如：

```yaml
repo_path: /mnt/xds/mtp/spec_train/speculators
container_repo_path: /mnt/xds/mtp/spec_train/speculators
```

`create` 模式不带 `--rm`，与给出的手工命令一致。停止后的同名容器仍会
保留；再次创建前需要人工确认并删除旧容器，或者切换到 `existing` 模式。

在 `existing` 模式下，manager 的 `stop` 会先按照 PID 文件优雅停止本集群
记录的 verifier、trainer 和 smoke 任务，然后在所有配置机器上分别对复用容器
执行一次 `docker restart -t 30`。容器不会被删除，但其中残留的 NPU worker
和进程内状态会被清理。因此不要让与本训练无关的服务共用这些容器。
manager 会先向任务 PID 发送 SIGTERM，但默认只等待 15 秒，因为紧随其后的
容器重启才是最终清理边界；可通过 `STOP_GRACE_SECONDS=N` 调整等待时间。

`existing` 模式还支持按角色清理：`restart-verifiers` 只停止 verifier 任务并
重启四个 verifier 容器；`restart-trainers` 同时停止 trainer/smoke 任务，但
只对四个共用的 trainer 容器各重启一次。执行 `restart-verifiers` 后需要再次
运行 `start-verifiers` 才会重新加载模型。

### 容器内实际执行什么

不需要人工进入任何容器操作：

- verifier、trainer 和 smoke 均不执行 pip install。启动脚本把
  `/mnt/xds/mtp/spec_train/speculators/src` 和
  `/mnt/xds/mtp/spec_train/speculators/hs_connectors/src` 加入
  `PYTHONPATH`；trainer 随后直接进入两机 32-rank 或四机 64-rank `torchrun`。

这是必需的兼容策略：镜像内 vLLM 0.23 要求 `setuptools<81`，而仓库的
editable build isolation 要求 `setuptools>=82`。不要为了 editable 安装升级
容器内 setuptools。只有自定义镜像已解决该版本约束时，才可以显式打开
`install_speculators_trainer`。
- 所有配置机器统一使用 YAML 中的同一个 vLLM-Ascend 镜像。

trainer 首次加载 verifier 的 embedding/head 时，会使用每台容器本地的
`/tmp/speculators-glm52-verifier-weights.lock` 将16个rank串行化。第一个rank
从SFS读取并预热本机页缓存，后续rank依次加载，避免所有进程同时陷入
`D/lock_page`。持锁进程退出时内核会自动释放该建议锁。

因此你的理解基本正确，但 trainer 也不需要手工进入容器执行 pip，包装脚本会
自动完成。只有使用自制镜像且已经预装代码时，才建议把 trainer 开关改为
`false`。

### 转换约 120 万条 Nuoya 数据

转换脚本会逐个处理 2K/8K 目录中的 JSONL，完成一个分片就写入完成标记；
中断重跑时会复用已完成分片，最后原子发布一份打乱后的 32K Hugging Face
Arrow 数据集：

```bash
cd /mnt/xds/mtp/spec_train/speculators
nohup python scripts/prepare_glm52_nuoya_32k.py \
  --model /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4 \
  > /mnt/xds/mtp/spec_train/dataset/prepare-nuoya-32k.log 2>&1 &
```

默认输出为：

```text
/mnt/xds/mtp/spec_train/dataset/hf/nuoya-average2k8k-32k
```

输出中的 `conversion_manifest.json` 会记录最终条数、平均长度、P50/P90/P99、
总 token 数，以及 GLM-5.2 单层 BF16 hidden-state 缓存的预计容量。

### 8K 小规模混合数据配置

仓库还提供一套不覆盖上述 32K 数据的 8K 配置。它严格选择按路径排序后的
Nuoya 前 5 个 JSONL，以及 `average-8k` 目录中的第 1 个 JSONL：

```bash
cd /mnt/xds/mtp/spec_train/speculators
nohup bash examples/train/prepare_glm52_nuoya_8k.sh \
  > /mnt/xds/mtp/spec_train/dataset/prepare-nuoya-first5-long1-8k.log 2>&1 &
```

默认输出为：

```text
/mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-8k
```

实际选中的 6 个源文件会写入输出目录的 `conversion_manifest.json`。如源目录
不足 5 个或 1 个 JSONL，脚本会直接失败，不会悄悄生成不完整数据。

复制独立的 8K 集群模板并填写 IP、容器模式和 smoke ID：

```bash
cp examples/train/mtp_glm52_ascend_online_4v4t_8k.example.yaml \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t-8k.yaml
```

该模板使用以下长度预算：

```yaml
data_path: /mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-8k
verifier_max_model_len: 8193
verifier_max_batched_tokens: 8208
verifier_gpu_memory_utilization: 0.90
total_seq_len: 8192
smoke_seq_len: 8192
```

`8193` 给 hidden-state 请求保留 1 个生成 token，TP16 再向上补齐到 `8208`。
8K 模板使用独立的 hidden-state、checkpoint 和日志目录，避免命中此前不同数据
或长度产生的缓存。切换配置后必须重启 verifier 才能让新的模型长度生效。

需要注意：8K 会显著降低 verifier KV cache 和训练 forward/backward 激活内存，
但不会降低 `fully_shard()` 把完整 MoE 参数搬上 NPU 时的初始化峰值。如果仍在
`_move_states_to_device` 申请 6 GiB 处 OOM，应先清理旧 NPU 进程，或继续改造
FSDP 的 CPU/meta 分片加载流程。

### 4K 显存安全配置

如果 8K 在 FSDP forward/backward all-gather 阶段仍需额外申请 6 GiB 而 OOM，
使用独立的 4K 数据和缓存配置。准备脚本仍严格选择 Nuoya 前 5 个 JSONL 和
`average-8k` 的第 1 个 JSONL，但按 4096 tokens 重新截断和打包：

```bash
cd /mnt/xds/mtp/spec_train/speculators
nohup bash examples/train/prepare_glm52_nuoya_4k.sh \
  > /mnt/xds/mtp/spec_train/dataset/prepare-nuoya-first5-long1-4k.log 2>&1 &
```

输出目录为：

```text
/mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-4k
```

复制专用于 existing 容器的配置，并填写 8 个 IP 与唯一 smoke ID：

```bash
cp examples/train/mtp_glm52_ascend_online_4v4t_4k.example.yaml \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t-4k.yaml
```

该配置使用独立的 `hidden_states_path`、checkpoint 和日志目录，核心限制为：

```yaml
verifier_max_model_len: 4097
verifier_max_num_seqs: 4
verifier_max_batched_tokens: 8224
total_seq_len: 4096
smoke_seq_len: 4096
fsdp_wrap_policy: memory_efficient
fsdp_experts_per_unit: 8
mtp_logits_chunk_size: 256
mtp_activation_checkpointing: true
```

这里不是只靠缩短序列省显存：完整 `lm_head` 会作为独立 FSDP 分组在使用后
重新分片，三步 MTP 的所有 logits chunk 共用一次 head 生命周期；MTP 层激活在
反向时重算。启动日志中的 `FSDP group` 会同时显示 FP32 master 大小和 BF16
all-gather 大小，便于区分参数峰值与序列激活峰值。memory-efficient 策略还会用
BF16 做梯度归约、禁用 backward prefetch，并让 root group 在 forward 后重新分片，
避免 6 GiB all-gather 与上一组 FP32 梯度通信缓冲重叠。

GLM routed experts 原始格式把 256 个 experts 融合进两个总计约 18 GiB 的 3-D
Parameter。FULL_SHARD 仍需在每次计算前恢复完整 Parameter，不能仅靠 wrapper 自动
拆分。`fsdp_experts_per_unit: 8` 会在 FSDP 前沿 expert 维度创建 32 个独立计算单元，
保持 top-k 路由和梯度等价，同时把最大 expert all-gather 降至约 0.56 GiB。启动时
应看到 `FSDP GLM expert chunking: ... units=32 experts_per_unit=8`，且不应再出现
`module=mtp_layers.0.mlp.experts ... all_gather_size_gib=18.00`。

在线训练会保留 YAML 中的全部 verifier endpoint。file backend 在 shared hidden
states 目录用原子 lease 汇总每台 verifier 的在途 token 数，新请求选择 token
负载最低的 endpoint；该机制不依赖 NFS `flock`。连接失败时本次请求直接切换
下一台，不再把某个 local rank 永久绑定到一台 verifier。若共享协调不可用则退化
为本地轮转。4K 模板允许最多四个短 prompt，但 token budget 把完整 4K prompt
限制为同时两个；继续提高 token budget 前应先检查 NPU 峰值和共享存储吞吐。
每个完成样本还会输出 `VERIFIER_REQUEST`，分别记录服务请求与共享文件读取耗时，
从而直接判断慢在 verifier 计算还是 NFS。

从 8K 切换到上述 4K verifier 限制后，必须重启 verifier；仅修改 trainer 的
`total_seq_len` 而继续使用 8K verifier 限制时则无需重启。完整切换命令：

```bash
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
CONFIG=/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t-4k.yaml
bash "$MANAGER" restart-verifiers --config "$CONFIG"
bash "$MANAGER" start-verifiers --config "$CONFIG"
bash "$MANAGER" wait-verifiers --config "$CONFIG"
bash "$MANAGER" restart-trainers --config "$CONFIG"
bash "$MANAGER" smoke --config "$CONFIG"
```

## 3. 一键预检

```bash
cd /mnt/xds/mtp/spec_train/speculators
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
CONFIG=/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
bash "$MANAGER" preflight --config "$CONFIG"
```

预检会在所有配置机器上检查模型配置、tokenizer、数据集、共享缓存路径、NPU
数量以及关键 Python/vLLM 环境。任意节点失败时不要启动正式训练。

## 4. 启动 verifier

```bash
bash "$MANAGER" start-verifiers --config "$CONFIG"
bash "$MANAGER" wait-verifiers --config "$CONFIG"
```

`start-verifiers` 是幂等操作：逐台检查 `/health`，健康节点直接跳过；进程
仍在运行但尚未健康的节点也不会重复启动；只有确认任务未运行的节点才会
拉起。只处理一台时使用：

```bash
bash "$MANAGER" start-verifier --index 2 --config "$CONFIG"
```

这里的 `--index` 对应 YAML 中 `verifier_ips` 的下标 `0..3`。

`wait-verifiers` 会等待四个 `/health` 接口全部返回 HTTP 200。启动日志在：

```text
/mnt/xds/mtp/spec_train/logs/orchestrator/<容器名>.host.log
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/verifier0/verifier.log
...
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/verifier3/verifier.log
```

启动 verifier 时会自动执行
`scripts/patch_vllm_glm52_final_hidden_state.py` 和
`scripts/patch_vllm_ascend_hidden_state_cache.py`。前者解决 vLLM 0.23 的
DeepSeek/GLM forward 只在进入 decoder block 前采集辅助状态，循环层号为
`0..77`；MTP 所需的层号 `78` 表示最后一个 block 后经过 final norm 的输出。
后者避免 Ascend MLA cache 合并逻辑把 `HiddenStateCacheSpec` 当作含
`scale_dim` 的量化 KV cache。补丁不修改权重或模型配置，并且可重复执行。原文件备份为
`deepseek_v2.py.before-glm-final-aux-hidden-state-fix`。如需人工检查或恢复：

```bash
python scripts/patch_vllm_glm52_final_hidden_state.py --check
python scripts/patch_vllm_glm52_final_hidden_state.py --restore
python scripts/patch_vllm_ascend_hidden_state_cache.py --check
python scripts/patch_vllm_ascend_hidden_state_cache.py --restore
```

verifier 必须使用：

```text
VERIFIER_QUANTIZATION_MODE=ascend
VERIFIER_MODEL_PATH=.../v1-ascend-modelslim-v4
```

## 5. Smoke 测试

四个 verifier 健康后执行：

```bash
bash "$MANAGER" smoke
bash "$MANAGER" status
```

smoke 会按训练 world size 自动准备足够样本（64-rank 时至少 256 条）、训练
两步，并删除临时生成的 hidden states，不会污染正式缓存。smoke 会把
`log_freq` 自动设为 1，因此两步都会记录 step/loss。默认 smoke 长度为 1024，
也可用 YAML 的 `smoke_seq_len` 覆盖；8K 模板将它设为 8192。必须确认所有
trainer 的 smoke 容器都正常退出，且日志中没有：

```text
Traceback
ERROR
NaN
HCCL timeout
generation error
```

## 6. 正式在线训练

smoke 全部成功后启动：

```bash
bash "$MANAGER" train
```

不要把 epoch 1 单独启动成一次 `EPOCHS=1` 的任务。应当从开始就配置最终
epoch 数，例如 5，从而保持连续的 optimizer 和 cosine scheduler 状态。

默认缓存位置：

```text
/mnt/xds/mtp/spec_train/online_hidden_states/glm52-w4a8c8
```

默认 checkpoint：

```text
/mnt/xds/mtp/spec_train/checkpoints/glm52-w4a8c8-mtp3/
```

训练日志：

```text
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node0.log
...
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node3.log
```

每个 trainer 节点的 local-rank 0 会在模型初始化阶段每 30 秒记录一次结构化
心跳，例如：

```text
TRAIN_STARTUP phase=fsdp_shard status=heartbeat ... elapsed_seconds=90.0
TRAIN_STARTUP phase=initial_weight_sync status=skipped ...
TRAIN_STARTUP phase=startup_barrier status=heartbeat ...
TRAIN_STARTUP phase=optimizer_init status=started ...
TRAIN_STARTUP phase=first_batch status=heartbeat ...
```

心跳间隔和 fresh-run 快速路径可在集群 YAML 中配置：

```yaml
startup_heartbeat_seconds: 30
fsdp_skip_initial_broadcast: true
```

GLM-5.2 启动器保证所有 rank 都从同一个完整 `MTP_DRAFT_PATH` 加载权重，因此
默认跳过 FSDP 分片后的第二次 rank-0 全量 state-dict 广播，只保留最终 barrier。
这会减少启动时间和 rank 0 峰值内存。恢复已有 checkpoint 时该开关不会跳过
distributed checkpoint load；若要对比诊断，可临时设置为 `false` 恢复原行为。

## 7. epoch 1 之后严格离线恢复

如果在线训练的首批请求受共享存储写入影响，可以先独立完成整套 hidden-state
采集，再启动训练。每个 collector 使用对应 verifier，并按 verifier 数量将数据
切成互不重叠的连续分片；重复执行会跳过已经存在的
`hs_<index>.safetensors`：

```bash
CONFIG=/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t-8k.yaml
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh

bash "$MANAGER" start-verifiers --config "$CONFIG"
bash "$MANAGER" wait-verifiers --config "$CONFIG"
bash "$MANAGER" collect-offline --config "$CONFIG"
bash "$MANAGER" offline-status --config "$CONFIG"
```

Verifier 数量由 `verifier_ips` 的长度动态决定，不局限于 4 台。仓库提供了
8 台 verifier、无 trainer 的 8K 和 4K 专用采集模板：

```text
examples/train/mtp_glm52_ascend_offline_collect_8v.example.yaml
examples/train/mtp_glm52_ascend_offline_collect_8v_4k.example.yaml
```

复制并填写 8 个地址后，使用同样的命令启动。Manager 会自动向每个 collector
传递 `--world-size 8 --rank 0..7`。采集配置中的 `trainer_ips` 可以为空；
`smoke`、`train` 和 `offline` 命令仍要求改用包含 2 或 4 个 trainer 的训练 YAML。
采集 YAML 与训练 YAML 必须使用完全相同的 `data_path` 和
`hidden_states_path`。

4K 配置还提供了一个只封装上述 Manager 命令的便捷脚本。它不会启动 trainer，
八个 collector 仍按 rank 分配互不重叠的数据范围，并可重复执行来续传：

```bash
export CONFIG=/mnt/xds/mtp/spec_train/config/glm52-mtp3-offline-8v-4k.yaml
bash examples/train/collect_glm52_nuoya_4k_8v.sh prepare
bash examples/train/collect_glm52_nuoya_4k_8v.sh collect
bash examples/train/collect_glm52_nuoya_4k_8v.sh status
# 全部完成后：
bash examples/train/collect_glm52_nuoya_4k_8v.sh verify
```

全部 collector 结束后，先验证文件数量，并抽样加载 safetensors 检查 token 和
hidden-state 长度。只有检查通过才会原子写入 `.offline-ready.json`：

```bash
bash "$MANAGER" verify-offline --config "$CONFIG"
bash "$MANAGER" offline --config "$CONFIG"
```

`offline` 命令本身也会再次执行完整性检查，缓存不完整时不会启动任何 trainer。
采集中断时重新执行 `collect-offline` 即可续传；如需只停止 collector 而保留
verifier，执行：

```bash
bash "$MANAGER" stop-collectors --config "$CONFIG"
```

如果不准备等待完整数据集，可以从已经落盘的缓存生成一个可训练子数据集：

```bash
bash examples/train/prepare_glm52_partial_offline_4k.sh
```

脚本扫描 `hs_<原始索引>.safetensors`，筛选原始 HF 数据，并在输出中加入
`source_index`。它不会复制或重命名 hidden states。默认路径为：

```text
原始数据：/mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-4k
现有缓存：/mnt/xds/mtp/spec_train/hidden_states/glm52-w4a8-mg13-offline-4k
筛选数据：/mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-4k-partial-offline
```

运行纯离线训练时，把训练 YAML 的 `data_path` 改为筛选数据，
`hidden_states_path` 仍保持为现有缓存目录。输出目录已存在时脚本会拒绝覆盖；
需要重新生成时请通过 `OUTPUT_DATA=/new/path` 使用新目录。设置
`VALIDATE_SAMPLES=-1` 可以在发布筛选数据前逐个加载并校验所有缓存文件。

8K YAML 中相关参数为：

```yaml
offline_collection_concurrency: 1
offline_collection_max_samples: 0  # 0 表示完整数据集
offline_validation_samples: 8
```

可以把 `offline_collection_max_samples` 临时设为较小值验证链路，但
`verify-offline` 始终按完整数据集检查，因此试采集结果不会被误用于正式训练。
正式采集前恢复为 `0` 并再次执行 `collect-offline`。

当前 verifier 的 `max_num_seqs` 为 1，因此 collector 并发默认也是 1。提高并发
前应同步提高 verifier 的请求容量，并确认 NFS 写入没有成为瓶颈。

### 推荐的生产模式：本地生成、动态采集、纯离线训练

4K MTP3 建议直接使用：

```bash
cp examples/train/mtp_glm52_ascend_production_4v4t_4k.example.yaml \
  /mnt/xds/mtp/spec_train/config/mtp_glm52_production_4v4t_4k.yaml
# 填写 IP、容器名和 smoke_run_id 后：
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
CONFIG=/mnt/xds/mtp/spec_train/config/mtp_glm52_production_4v4t_4k.yaml
bash "$MANAGER" validate-config --config "$CONFIG"
bash "$MANAGER" production --config "$CONFIG"
```

`production` 会干净重启全部 verifier，确保新 connector 补丁和显存参数真正
生效；随后等待健康检查，启动动态 collector，等待并验证完整缓存，最后才启动
不访问 verifier 的 FSDP trainer。采集或训练失败后可重复执行，已经原子发布的
`hs_<index>.safetensors` 会被复用。

如果只需要确认在线生成、FSDP 前向、反向和 optimizer step 能否完整跑通，不必
等待全量离线缓存。下面的脚本会从已经填写的生产 YAML 自动生成隔离配置，停止
collector，重启 verifier，然后运行 1K、2-step 在线 smoke。请求文件被 trainer
读取后立即删除，不会写入生产缓存：

```bash
SOURCE_CONFIG=/mnt/xds/mtp/spec_train/config/mtp_glm52_production_4v4t_4k.yaml \
  bash examples/train/run_glm52_quick_online_smoke.sh run

# 观察状态
bash examples/train/run_glm52_quick_online_smoke.sh status

# 1K 成功后，用新的 run ID 验证 4K
SMOKE_SEQ_LEN=4096 SMOKE_RUN_ID=quick-4k-$(date +%Y%m%d-%H%M%S) \
  bash examples/train/run_glm52_quick_online_smoke.sh run
```

在线 smoke 的 verifier 与 trainer 位于不同主机，因此请求级 hidden state 仍需
短暂写入共享目录；`smoke` 固定传递 `--on-generate delete --force-generate`，加载
成功后立即删除文件。脚本使用带 run ID 的独立目录，既不读取也不删除生产离线
缓存。

该 profile 将 verifier 输出先写入每台机器的
`/tmp/speculators-glm52-hidden-states`。collector 校验 token/hidden-state 后，以
`.partial -> hs_<index>.safetensors` 原子发布到共享目录。这样 vLLM 的保存线程
不再被 NFS 延迟阻塞。动态目录 claim 允许空闲 verifier 接管慢节点尚未领取的
样本；请求并发和共享盘写并发分别由以下配置控制：

```yaml
verifier_staging_path: /tmp/speculators-glm52-hidden-states
offline_collection_schedule: dynamic
offline_collection_concurrency: 2
offline_collection_write_concurrency: 1
offline_collection_poll_interval: 30
```

训练侧使用 `mtp_training_strategy: sampled_step`。MTP3 仍然训练三个预测距离，
但每个 optimizer step 只为一个距离保留梯度图，按 step 0/1/2 轮换，并把该项
loss 乘以 3；因此它是原加权三步目标的均匀随机无偏估计。前置递归状态仍由
同一个 MTP 层按 teacher forcing 精确计算，只是不保存其 autograd 图。验证始终
使用完整三步展开。这里无偏的是 loss 值；梯度不会穿过前置递归 horizon，属于
截断反向传播。相较一次反向同时保留三个 GLM-MoE 图，这能显著降低峰值：

```yaml
mtp_training_strategy: sampled_step
mtp_activation_checkpointing: false
fsdp_experts_per_unit: 2
mtp_logits_chunk_size: 128
memory_log_freq: 1
```

这里 `total_seq_len: 4096` 是每 rank 的 packed token budget，collate 后张量形状
是 `[1, 4096, ...]`；它不是“4 个样本各 4096 token”。因此普通 batch-size
开关不能继续拆分一个 4K packed sequence。上述 sampled horizon、分块词表头、
细粒度 expert FSDP 和纯离线输入才是这条路径的主要显存/稳定性手段。

epoch 1 已完整遍历训练集和验证集后，可以使用：

```bash
bash "$MANAGER" offline
```

该模式：

- 自动恢复最新的数字 checkpoint；
- 不传 verifier endpoint；
- 不生成新的 hidden states；
- 遇到缓存缺失立即失败。

确认所有 trainer 都能稳定读取 batch 后，才可以停止 verifier。

在线和离线阶段之间不得修改：

```text
DATA_PATH
TRAIN_DATA_RATIO
TOTAL_SEQ_LEN
HIDDEN_STATES_PATH
OUTPUT_PATH
RUN_NAME
```

## 8. 状态、TensorBoard 和停止

### 控制节点 Web 看板

YAML 中加入以下配置后，manager 会在执行 `start-verifiers`、`smoke`、
`train` 或 `offline` 后，在控制节点自动启动只读看板：

```yaml
dashboard_host: 0.0.0.0
dashboard_port: 6007
dashboard_auto_start: true
```

浏览器访问 `http://<控制节点IP>:6007`。页面每 5 秒刷新一次，汇总共享的
host-wrapper 日志、verifier 详细日志和 `/health` 探测，展示全部 verifier 与
trainer 的阶段、epoch/step/loss、prompt/generation 吞吐、请求排队、KV cache、
日志更新时间及最近错误。初始化期间还会直接展示 FSDP 分片、权重同步、节点
barrier、优化器初始化和首个 batch 等待的阶段与耗时。它只在控制节点运行，
不会在八个业务容器里安装任何
依赖；对 `7.x` 内网 verifier 的健康探测会明确绕过 `HTTP_PROXY/HTTPS_PROXY`。

手动管理命令：

```bash
bash "$MANAGER" dashboard --config "$CONFIG"
bash "$MANAGER" dashboard-status --config "$CONFIG"
bash "$MANAGER" stop-dashboard --config "$CONFIG"
```

控制节点有多张网卡且打印出的地址不正确时，设置
`DASHBOARD_ADVERTISE_HOST=<控制节点IP>`。看板本身不带鉴权；非可信网络中应将
`dashboard_host` 设为 `127.0.0.1` 并通过 SSH 隧道访问，或用防火墙保护 6007
端口。集群 `stop` 不会停止看板，以便故障后继续查看最终日志。

查看所有配置机器相关容器和最近日志：

```bash
bash "$MANAGER" status
```

TensorBoard：

```bash
tensorboard \
  --logdir /mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/metrics \
  --host 0.0.0.0 --port 6006
```

检查缓存：

```bash
find /mnt/xds/mtp/spec_train/online_hidden_states/glm52-w4a8c8 \
  -type f -name 'hs_*.safetensors' | wc -l
du -sh /mnt/xds/mtp/spec_train/online_hidden_states/glm52-w4a8c8
```

停止当前容器名前缀对应的 verifier、trainer 和 smoke 容器：

```bash
bash "$MANAGER" stop
```

`stop` 不删除 checkpoint、缓存或日志。

## 9. MTP 推理补丁的边界

hidden-state launcher 会使用 `method=extract_hidden_states` 的 speculative
配置，但不会使用原生 `deepseek_mtp`。它只负责运行目标模型并生成 layer-78
hidden states，不会构造或加载原生 MTP drafter 的共享 embedding/head，因此
在线训练不需要 `patch_vllm_glm52_mtp_shared_weights.py`。

训练结束后，如果在 vLLM 中加载原生 MTP 做推理、接收率测试或 benchmark，
相应推理容器才需要应用共享 embedding/head loader 补丁。详情参见
[GLM-5.2 MG13 W4A8 ModelSlim 说明](glm52_mixed_compressed_tensors.md)。

## 10. 成功标准与恢复规则

verifier 启动时会自动应用 `scripts/patch_vllm_hidden_state_enolck.py` 和
`scripts/patch_vllm_hidden_state_connector_tp_gather.py`。
当 `/mnt/xds/mtp` 等共享文件系统不支持 `flock` 并返回 `Errno 37` 时，该补丁会
改为同步写完 hidden-state safetensors 后再返回路径。这样既不会使 EngineCore
崩溃，也不会让 trainer 读到未写完整的文件。补丁不修改模型权重或 config，
并且可以用以下命令检查或恢复：

```bash
python scripts/patch_vllm_hidden_state_enolck.py --check
python scripts/patch_vllm_hidden_state_enolck.py --restore
python scripts/patch_vllm_hidden_state_connector_tp_gather.py --check
python scripts/patch_vllm_hidden_state_connector_tp_gather.py --restore
```

verifier 会明确设置 `enable_dsa_cp=false`。当前 vLLM-Ascend 开启 DSA context
parallel 后，`ExampleHiddenStatesConnector` 只能看到本 worker 的序列分片；例如
CP=8 时，1024-token prompt 只会写出 128 行 hidden states，但 token_ids 仍是
完整 1024 个。在线训练仍以关闭 DSA CP 为主；保存边界补丁会额外使用完整
`token_ids` 长度校验结果，发现短分片时执行 TP gather，并在最终长度仍不一致
时直接拒绝写出文件。这样不会再把 128/1024 的坏文件静默交给 trainer。
TP gather 也会按各 rank 的真实 token 数移除对齐 padding；例如 TP16 下的
959-token 请求即使以 16 个 63-token 缓冲区 gather 成 1008 行，也会在验证
shard/副本布局后移除全局 token 流末尾的 49 行调度 padding，还原为 959 行。
所有 TP rank 仍会参加 gather，但只有 TP rank 0 执行 D2H 和文件发布；TP16
因此从每个请求 16 次重复保存降为 1 次，避免本地盘或 NFS 上的重复写和锁竞争。

成功标准：

- loss 为有限值；
- 没有 HCCL 或在线生成错误；
- epoch 1 训练和验证均完成；
- 已生成数字 checkpoint；
- epoch 1 缓存文件数量持续增加；
- epoch 2+ 缓存数量稳定且 verifier 请求停止增长。

恢复规则：

- 所有 trainer 应一起停止、一起恢复；
- 正常恢复使用最新数字 checkpoint；
- 不要未经检查就把 `interrupted` 目录当作完整 checkpoint；
- 任何训练或 verifier 仍在运行时，不要删除共享缓存；
- 发现缺少 `.ready` 的临时目录时，先确认没有生产进程，再移走诊断。
