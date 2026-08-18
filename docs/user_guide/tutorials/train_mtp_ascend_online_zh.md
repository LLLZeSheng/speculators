# 八台 Ascend 910C 在线训练 GLM-5.2 MTP3

本文是 4 台 verifier + 4 台 trainer 的生产操作说明。每台机器使用 16
张 NPU：verifier 采用 DP1 × TP16，四台 trainer 组成 64-rank FSDP 训练任务。

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
共享盘：/kos_ulan
代码：/kos_ulan/lzs/spec_train/speculators
量化 verifier：
  /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
原始 verifier 权重：
  /mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1
原生 MTP 初始化模型：
  /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
训练数据：/kos_ulan/lzs/spec_train/dataset/hf/nuoya-average2k8k-32k
```

MG13 主模型层仍然是只用于推理的 W4A8 权重，但准备好的 v4 运行视图中，
第 78 层原生 MTP 以及共享 embedding/lm_head 是浮点权重。因此该目录现在
也是首选的 `MTP_INIT_MODEL_PATH`。Speculators 会从 `mtp.safetensors`（或
ModelSlim 索引）读取 MTP，并在转换前拒绝任何整数量化的可训练权重；共享
权重通过 `quant_model_weights.safetensors.index.json` 定位。整个过程不会
修改原模型或运行视图。

所有节点应当具备：

- 相同的 `/kos_ulan` 挂载；
- 相同的代码路径；
- 可互通的训练网络；
- 控制节点到八台机器的免密 SSH；
- Docker 和 16 张可用 Ascend NPU。

## 2. 自己定义并维护 YAML

YAML 是你维护的唯一配置源。管理脚本只读取和校验，不会生成、覆盖或回写
YAML。先复制模板，然后直接填写机器、容器和路径信息：

```bash
cd /kos_ulan/lzs/spec_train/speculators
mkdir -p /kos_ulan/lzs/spec_train/config
cp examples/train/mtp_glm52_ascend_online_4v4t.example.yaml \
  /kos_ulan/lzs/spec_train/config/glm52-mtp3-4v4t.yaml
vim /kos_ulan/lzs/spec_train/config/glm52-mtp3-4v4t.yaml
```

需要自行填写或确认：

- `verifier_ips` 中四台 verifier 的 IP；
- `trainer_ips` 中四台 trainer 的 IP；
- `container_mode`、容器名称、镜像和挂载列表；
- 代码、模型、数据、hidden states、checkpoint 和日志路径；
- 所有 `FILL_*` 占位值必须被替换；
- `create` 模式的容器名会自动成为 `glm52-online-mtp3-verifier0`、
  `glm52-online-mtp3-trainer0` 等；`existing` 模式复用
  `existing_container_name`。

建议配置文件放在：

```text
/kos_ulan/lzs/spec_train/config/glm52-mtp3-4v4t.yaml
```

启动任何远端任务前先进行纯本地校验：

```bash
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
CONFIG=/kos_ulan/lzs/spec_train/config/glm52-mtp3-4v4t.yaml
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
container_repo_path: /kos_ulan/lzs/spec_train/speculators
repo_path: /kos_ulan/lzs/spec_train/speculators
container_mounts:
  - /mnt/xds/sfs:/mnt/xds/sfs
  - /kos_ulan:/kos_ulan
  - /kos_ulan/lzs/spec_train/speculators:/kos_ulan/lzs/spec_train/speculators
  - /root/.cache:/root/.cache
install_speculators_verifier: false
install_speculators_trainer: false
```

`container_mode: create` 不需要、也不应填写 `existing_container_name`。
它使用 `docker run` 创建容器，参数与标准 A3 启动方式对齐：host 网络、
默认 1 GiB shm、16 张 NPU、管理设备、Ascend 驱动、
`/mnt/xds/sfs`、`/kos_ulan` 和 `/root/.cache`。需要额外挂载目录时，直接在
`container_mounts` 追加 `宿主机路径:容器路径[:ro|rw]`。管理任务通过
nohup 后台执行，因此不会加入只能在交互终端使用的 `-it`。

如果八台机器上已经提前创建好同名容器，使用：

```yaml
container_mode: existing
existing_container_name: glm52-w4a8-mg13-speculator-training
```

此时管理器通过 `docker exec` 运行任务，`container_image`、
`container_shm_size` 和 `container_mounts` 不会重新应用；这些必须在原容器
创建时已经设置正确。`stop` 只根据 PID 文件停止本次 MTP 任务，不会执行
`docker stop`，因此不会关闭或删除已有容器。

如果已有容器正是用本文开头那种只挂载 `/kos_ulan:/kos_ulan` 的命令创建，
容器内并不存在 `/workspace/speculators` 映射。这时应把 YAML 改为实际可见
的代码目录，例如：

```yaml
repo_path: /kos_ulan/lzs/spec_train/speculators
container_repo_path: /kos_ulan/lzs/spec_train/speculators
```

`create` 模式不带 `--rm`，与给出的手工命令一致。停止后的同名容器仍会
保留；再次创建前需要人工确认并删除旧容器，或者切换到 `existing` 模式。

在 `existing` 模式下，manager 的 `stop` 会先按照 PID 文件优雅停止本集群
记录的 verifier、trainer 和 smoke 任务，然后在八台机器上分别对复用容器
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
  `/kos_ulan/lzs/spec_train/speculators/src` 和
  `/kos_ulan/lzs/spec_train/speculators/hs_connectors/src` 加入
  `PYTHONPATH`；trainer 随后直接进入四机 64-rank `torchrun`。

这是必需的兼容策略：镜像内 vLLM 0.23 要求 `setuptools<81`，而仓库的
editable build isolation 要求 `setuptools>=82`。不要为了 editable 安装升级
容器内 setuptools。只有自定义镜像已解决该版本约束时，才可以显式打开
`install_speculators_trainer`。
- 八台机器统一使用 YAML 中的同一个 vLLM-Ascend 镜像。

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
cd /kos_ulan/lzs/spec_train/speculators
nohup python scripts/prepare_glm52_nuoya_32k.py \
  --model /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4 \
  > /kos_ulan/lzs/spec_train/dataset/prepare-nuoya-32k.log 2>&1 &
```

默认输出为：

```text
/kos_ulan/lzs/spec_train/dataset/hf/nuoya-average2k8k-32k
```

输出中的 `conversion_manifest.json` 会记录最终条数、平均长度、P50/P90/P99、
总 token 数，以及 GLM-5.2 单层 BF16 hidden-state 缓存的预计容量。

## 3. 一键预检

```bash
cd /kos_ulan/lzs/spec_train/speculators
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
CONFIG=/kos_ulan/lzs/spec_train/config/glm52-mtp3-4v4t.yaml
bash "$MANAGER" preflight --config "$CONFIG"
```

预检会在八台机器上检查模型配置、tokenizer、数据集、共享缓存路径、NPU
数量以及关键 Python/vLLM 环境。任意节点失败时不要启动正式训练。

## 4. 启动 verifier

```bash
bash "$MANAGER" start-verifiers --config "$CONFIG"
bash "$MANAGER" wait-verifiers --config "$CONFIG"
```

`wait-verifiers` 会等待四个 `/health` 接口全部返回 HTTP 200。启动日志在：

```text
/kos_ulan/spec_train/logs/orchestrator/<容器名>.host.log
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier0/verifier.log
...
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier3/verifier.log
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

smoke 使用 64 条样本、训练两步，并删除临时生成的 hidden states，不会污染
正式缓存。必须确认四台 trainer 的 smoke 容器都正常退出，且日志中没有：

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
/kos_ulan/spec_train/online_hidden_states/glm52-w4a8c8
```

默认 checkpoint：

```text
/kos_ulan/spec_train/checkpoints/glm52-w4a8c8-mtp3/
```

训练日志：

```text
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node0.log
...
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node3.log
```

## 7. epoch 1 之后严格离线恢复

epoch 1 已完整遍历训练集和验证集后，可以使用：

```bash
bash "$MANAGER" offline
```

该模式：

- 自动恢复最新的数字 checkpoint；
- 不传 verifier endpoint；
- 不生成新的 hidden states；
- 遇到缓存缺失立即失败。

确认四台 trainer 都能稳定读取 batch 后，才可以停止 verifier。

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

查看八台机器相关容器和最近日志：

```bash
bash "$MANAGER" status
```

TensorBoard：

```bash
tensorboard \
  --logdir /kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/metrics \
  --host 0.0.0.0 --port 6006
```

检查缓存：

```bash
find /kos_ulan/spec_train/online_hidden_states/glm52-w4a8c8 \
  -type f -name 'hs_*.safetensors' | wc -l
du -sh /kos_ulan/spec_train/online_hidden_states/glm52-w4a8c8
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

verifier 启动时会自动应用 `scripts/patch_vllm_hidden_state_enolck.py`。
当 `/kos_ulan` 等共享文件系统不支持 `flock` 并返回 `Errno 37` 时，该补丁会
改为同步写完 hidden-state safetensors 后再返回路径。这样既不会使 EngineCore
崩溃，也不会让 trainer 读到未写完整的文件。补丁不修改模型权重或 config，
并且可以用以下命令检查或恢复：

```bash
python scripts/patch_vllm_hidden_state_enolck.py --check
python scripts/patch_vllm_hidden_state_enolck.py --restore
```

verifier 会明确设置 `enable_dsa_cp=false`。当前 vLLM-Ascend 开启 DSA context
parallel 后，`ExampleHiddenStatesConnector` 只能看到本 worker 的序列分片；例如
CP=8 时，1024-token prompt 只会写出 128 行 hidden states，但 token_ids 仍是
完整 1024 个。connector 尚未实现 CP all-gather，因此在线训练必须关闭 DSA
CP。这会降低 verifier 吞吐，但能保证训练数据长度正确。

成功标准：

- loss 为有限值；
- 没有 HCCL 或在线生成错误；
- epoch 1 训练和验证均完成；
- 已生成数字 checkpoint；
- epoch 1 缓存文件数量持续增加；
- epoch 2+ 缓存数量稳定且 verifier 请求停止增长。

恢复规则：

- 四台 trainer 应一起停止、一起恢复；
- 正常恢复使用最新数字 checkpoint；
- 不要未经检查就把 `interrupted` 目录当作完整 checkpoint；
- 任何训练或 verifier 仍在运行时，不要删除共享缓存；
- 发现缺少 `.ready` 的临时目录时，先确认没有生产进程，再移走诊断。
