# 八台 Ascend 910C 在线训练 GLM-5.2 MTP3

本文是 4 台 verifier + 4 台 trainer 的生产操作说明。每台机器使用 16
张 NPU：verifier 采用 DP2 × TP8，四台 trainer 组成 64-rank FSDP 训练任务。

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
代码：/kos_ulan/spec_train/speculators
量化 verifier：
  /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
原始 verifier 权重：
  /mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1
BF16 初始化模型：/kos_ulan/models/GLM-5.2
训练数据：/kos_ulan/datasets/glm52-mtp-online
```

BF16 初始化模型必须包含原生 MTP layer。W4A8 verifier 只用于推理和生成
hidden states，不能作为可训练 MTP 权重的初始化来源。

所有节点应当具备：

- 相同的 `/kos_ulan` 挂载；
- 相同的代码路径；
- 可互通的训练网络；
- 控制节点到八台机器的免密 SSH；
- Docker 和 16 张可用 Ascend NPU。

## 2. 只填写 IP 和容器名前缀

在任意一台能够 SSH 到其他节点、且挂载 `/kos_ulan` 的控制节点执行：

```bash
cd /kos_ulan/spec_train/speculators

bash examples/train/manage_mtp_glm52_ascend_online_4v4t.sh configure \
  --verifier-ips V0,V1,V2,V3 \
  --trainer-ips T0,T1,T2,T3 \
  --container-prefix glm52-online-mtp3
```

其中：

- `V0～V3` 是四台 verifier 的 IP；
- `T0～T3` 是四台 trainer 的 IP；
- `glm52-online-mtp3` 是容器名前缀；
- 最终容器名会自动成为 `glm52-online-mtp3-verifier0`、
  `glm52-online-mtp3-trainer0` 等。

生成的共享配置位于：

```text
/kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
```

如果 BF16 模型、数据或代码不在默认位置，只需在首次配置时增加：

```bash
  --mtp-init-model /kos_ulan/实际模型路径 \
  --data-path /kos_ulan/实际数据路径 \
  --repo-path /kos_ulan/实际代码路径
```

脚本会根据每台机器的 IP 自动识别网卡，不要求八台机器使用相同的网卡名。
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

## 3. 一键预检

```bash
cd /kos_ulan/spec_train/speculators
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
bash "$MANAGER" preflight
```

预检会在八台机器上检查模型配置、tokenizer、数据集、共享缓存路径、NPU
数量以及关键 Python/vLLM 环境。任意节点失败时不要启动正式训练。

## 4. 启动 verifier

```bash
bash "$MANAGER" start-verifiers
bash "$MANAGER" wait-verifiers
```

`wait-verifiers` 会等待四个 `/health` 接口全部返回 HTTP 200。启动日志在：

```text
/kos_ulan/spec_train/logs/orchestrator/<容器名>.host.log
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier0/verifier.log
...
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier3/verifier.log
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

本在线训练 verifier 没有传入 `--speculative-config`。它只负责运行目标模型并
生成 layer-78 hidden states，不会构造 MTP drafter，因此在线训练不需要
`patch_vllm_glm52_mtp_shared_weights.py`。

训练结束后，如果在 vLLM 中加载原生 MTP 做推理、接收率测试或 benchmark，
相应推理容器才需要应用共享 embedding/head loader 补丁。详情参见
[GLM-5.2 MG13 W4A8 ModelSlim 说明](glm52_mixed_compressed_tensors.md)。

## 10. 成功标准与恢复规则

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
