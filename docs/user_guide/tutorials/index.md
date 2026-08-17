# Tutorials

Step-by-step tutorials to guide you through complete workflows, from data preparation to serving trained models in production.

## [Serve in vLLM](serve_vllm.md)

Deploy your trained speculator models in vLLM for production inference.

**Time required:** ~5 minutes

## [Train Eagle-3 Model Online](train_eagle3_online.md)

Learn how to train an Eagle-3 speculator using online training, where hidden states are generated on-demand during training.

**Time required:** ~30 mins

## [Train Eagle-3 Model Offline](train_eagle3_offline.md)

Learn how to train an Eagle-3 speculator using offline training with pre-generated hidden states.

**Time required:** ~3 hours

## [Train DFlash Model Online](train_dflash_online.md)

Learn how to train a DFlash speculator model with block-based token generation.

**Time required:** ~25 mins

## [Train P-eagle Model offline](train_peagle_offline.md)

Learn how to train a P-eagle speculator model with COD sampling.

**Time required:** ~50 mins

## [Train MTP Model Online](train_mtp_online.md)

Learn how to finetune a model's native MTP head on domain-specific data using online training.

**Time required:** ~8 mins for Qwen3.5-9B on 2x H200 GPUs (varies by model size)

## [Train GLM-5.2 MTP3 on Eight Ascend Nodes](train_mtp_ascend_online.md)

Run a four-verifier plus four-trainer job whose first epoch caches online
hidden states and whose later epochs reuse them like offline training.

## [八台 Ascend 910C 在线训练 GLM-5.2 MTP3](train_mtp_ascend_online_zh.md)

中文版生产说明：只填写八台机器 IP 和容器名前缀，统一完成预检、verifier、
smoke、在线训练、离线恢复、状态检查与停止。

## [GLM-5.2 MG13 W4A8 on vLLM Ascend](glm52_mixed_compressed_tensors.md)

Prepare a non-destructive ModelSlim runtime view, including native MTP metadata.

## [Response Regeneration](response_regeneration.md)

Regenerate dataset responses using your target model for improved drafter alignment.

**Time required:** ~10 minutes

## [Evaluating Model Performance](evaluating_performance.md)

Benchmark and evaluate your trained speculator models.
