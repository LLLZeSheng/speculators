"""MTP checkpoint converter.

Extracts only the MTP layer weights from a checkpoint with native MTP
layers and saves a Speculators checkpoint that loads with
``MTPDraftModel.from_pretrained(path)``.

Only the native MTP subtree is extracted from the (potentially sharded)
safetensors file; the rest of the model is never loaded. Qwen checkpoints use
``mtp.*`` while GLM MoE DSA checkpoints use the extra transformer layer at
``model.layers.<num_hidden_layers>.*``. The embed_tokens and lm_head are loaded
from the verifier at runtime via ``load_verifier_weights()``.
"""

import json
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
)
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import list_checkpoint_keys

__all__ = [
    "MTP_EXACT_REMAP",
    "MTP_PREFIX_REMAP",
    "MTPConverter",
    "remap_mtp_key_to_native",
]

_MTP_PREFIX = "mtp."

MTP_EXACT_REMAP: dict[str, str] = {
    "mtp.fc.weight": "mtp_layers.0.input_proj.weight",
    "mtp.norm.weight": "mtp_layers.0.final_layernorm.weight",
}

MTP_PREFIX_REMAP: list[tuple[str, str]] = [
    ("mtp.pre_fc_norm_hidden.", "mtp_layers.0.hidden_layernorm."),
    ("mtp.pre_fc_norm_embedding.", "mtp_layers.0.token_layernorm."),
    ("mtp.layers.0.", "mtp_layers.0."),
]

_INVERSE_MTP_EXACT_REMAP = {v: k for k, v in MTP_EXACT_REMAP.items()}
_INVERSE_MTP_PREFIX_REMAP = [(dst, src) for src, dst in MTP_PREFIX_REMAP]

_GLM_MTP_EXACT_SUFFIX_REMAP: dict[str, str] = {
    "eh_proj.weight": "mtp_layers.0.input_proj.weight",
    "hnorm.weight": "mtp_layers.0.hidden_layernorm.weight",
    "enorm.weight": "mtp_layers.0.token_layernorm.weight",
    "shared_head.norm.weight": "mtp_layers.0.final_layernorm.weight",
}
_INVERSE_GLM_MTP_EXACT_SUFFIX_REMAP = {
    value: key for key, value in _GLM_MTP_EXACT_SUFFIX_REMAP.items()
}

_EXPERT_PATTERN = re.compile(
    r"^(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def remap_mtp_key_to_native(
    key: str,
    model_type: str,
    num_hidden_layers: int,
) -> str:
    """Map a Speculators MTP key back to a verifier-native checkpoint key."""
    if model_type == "glm_moe_dsa":
        prefix = f"model.layers.{num_hidden_layers}."
        if key in _INVERSE_GLM_MTP_EXACT_SUFFIX_REMAP:
            return prefix + _INVERSE_GLM_MTP_EXACT_SUFFIX_REMAP[key]
        generic_prefix = "mtp_layers.0."
        if key.startswith(generic_prefix):
            return prefix + key[len(generic_prefix) :]
        return key

    if key in _INVERSE_MTP_EXACT_REMAP:
        return _INVERSE_MTP_EXACT_REMAP[key]
    for source, destination in _INVERSE_MTP_PREFIX_REMAP:
        if key.startswith(source):
            return destination + key[len(source) :]
    return key


class MTPConverter:
    """Extract the MTP head from a checkpoint with native MTP layers.

    Reads only the MTP layer, embed_tokens, and lm_head from the source
    checkpoint.  Sharded safetensors files are handled transparently via
    the weight index -- the main transformer stack is never loaded.
    """

    @staticmethod
    def validate_load_result(missing: list[str], unexpected: list[str]) -> None:
        """Reject checkpoint/model drift while allowing runtime-only weights."""
        if unexpected:
            raise ValueError(
                "Unexpected keys in extracted weights -- the checkpoint structure "
                "does not match the model architecture. "
                f"Unexpected keys: {unexpected}"
            )
        critical_missing = [key for key in missing if key.startswith("mtp_layers.")]
        if critical_missing:
            raise ValueError(
                "Critical MTP layer weights missing after extraction. The checkpoint "
                "may be incompatible or use an unsupported format. "
                f"Missing keys: {critical_missing}"
            )

    def convert_to_state_dict(
        self,
        input_path: str | Path,
        cache_dir: str | Path | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract native MTP weights and return them as a state dict.

        Performs the full pipeline (download/locate → verify → extract →
        remap → fuse MoE experts) without writing anything to disk.
        """
        logger.info(f"Extracting native MTP weights from {input_path}")

        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source_config = load_checkpoint_config(local_path)
        if "text_config" in source_config:
            source_config = source_config["text_config"]
        all_keys = list_checkpoint_keys(local_path)
        source_prefix = self._resolve_mtp_prefix(all_keys, source_config)

        weights = self._extract_weights(local_path, all_keys, source_prefix)
        weights = self._fuse_moe_experts(weights)
        logger.info(f"Extracted {len(weights)} MTP weight tensors")
        return weights

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        num_speculative_steps: int = 3,
        validate: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        logger.info(f"Converting MTP checkpoint: {input_path}")

        weights = self.convert_to_state_dict(input_path, cache_dir)

        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source_config = load_checkpoint_config(local_path)

        if "text_config" in source_config:
            source_config = source_config["text_config"]

        config = self._build_config(source_config, base_model, num_speculative_steps)
        saved_path = self._save(config, weights, output_path)
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate(saved_path)

    @staticmethod
    def _resolve_mtp_prefix(keys: list[str], source_config: dict) -> str:
        if any(key.startswith(_MTP_PREFIX) for key in keys):
            return _MTP_PREFIX

        if source_config.get("model_type") == "glm_moe_dsa":
            layer_idx = source_config.get("num_hidden_layers")
            prefix = f"model.layers.{layer_idx}."
            if any(key.startswith(prefix) for key in keys):
                return prefix
            raise ValueError(
                f"GLM checkpoint declares native MTP but no '{prefix}' "
                "weights were found."
            )

        raise ValueError(
            f"No keys with prefix '{_MTP_PREFIX}' found. This converter "
            "requires a checkpoint with supported native MTP layers."
        )

    @staticmethod
    def _remap_key(key: str, source_prefix: str = _MTP_PREFIX) -> str:
        if source_prefix != _MTP_PREFIX and key.startswith(source_prefix):
            suffix = key[len(source_prefix) :]
            if suffix in _GLM_MTP_EXACT_SUFFIX_REMAP:
                return _GLM_MTP_EXACT_SUFFIX_REMAP[suffix]
            return "mtp_layers.0." + suffix
        if key in MTP_EXACT_REMAP:
            return MTP_EXACT_REMAP[key]
        for src, dst in MTP_PREFIX_REMAP:
            if key.startswith(src):
                return dst + key[len(src) :]
        return key

    @staticmethod
    def _fuse_moe_experts(
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse individual expert weights into packed tensors.

        MoE checkpoints (e.g. Qwen3-Next) store per-expert weights as
        separate keys (``experts.{N}.gate_proj``), but the transformers
        model uses fused 3D tensors (``experts.gate_up_proj``).
        """
        groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        non_expert: dict[str, torch.Tensor] = {}

        for key, tensor in weights.items():
            m = _EXPERT_PATTERN.match(key)
            if m:
                prefix = m.group(1)
                idx = int(m.group(2))
                proj = m.group(3)
                groups.setdefault(prefix, {}).setdefault(idx, {})[proj] = tensor
            else:
                non_expert[key] = tensor

        if not groups:
            return weights

        for prefix, experts_by_idx in groups.items():
            num_experts = max(experts_by_idx.keys()) + 1
            expected = set(range(num_experts))
            if set(experts_by_idx.keys()) != expected:
                raise ValueError(
                    f"Non-contiguous expert indices at {prefix}: "
                    f"found {sorted(experts_by_idx.keys())}, "
                    f"expected 0..{num_experts - 1}"
                )

            for i in range(num_experts):
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    if proj_name not in experts_by_idx[i]:
                        raise ValueError(
                            f"Expert {i} at {prefix} missing {proj_name} weight"
                        )

            gate_list = [experts_by_idx[i]["gate_proj"] for i in range(num_experts)]
            up_list = [experts_by_idx[i]["up_proj"] for i in range(num_experts)]
            down_list = [experts_by_idx[i]["down_proj"] for i in range(num_experts)]

            gate_up = torch.stack(
                [
                    torch.cat([g, u], dim=0)
                    for g, u in zip(gate_list, up_list, strict=True)
                ]
            )
            down = torch.stack(down_list)

            non_expert[f"{prefix}.gate_up_proj"] = gate_up
            non_expert[f"{prefix}.down_proj"] = down
            logger.debug(
                f"Fused {num_experts} experts at {prefix}: "
                f"gate_up_proj={gate_up.shape}, "
                f"down_proj={down.shape}"
            )

        return non_expert

    def _extract_weights(
        self,
        checkpoint_dir: Path,
        all_keys: list[str],
        source_prefix: str,
    ) -> dict[str, torch.Tensor]:
        needed = {k for k in all_keys if k.startswith(source_prefix)}

        index_path = checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            return self._extract_from_shards(
                checkpoint_dir, index_path, needed, source_prefix
            )

        single = checkpoint_dir / "model.safetensors"
        if single.exists():
            weights: dict[str, torch.Tensor] = {}
            with safe_open(str(single), framework="pt") as f:
                for key in needed & set(f.keys()):
                    weights[self._remap_key(key, source_prefix)] = f.get_tensor(key)
            return weights

        raise FileNotFoundError(
            f"No safetensors checkpoint found at {checkpoint_dir}. "
            "Expected model.safetensors.index.json or model.safetensors."
        )

    def _extract_from_shards(
        self,
        checkpoint_dir: Path,
        index_path: Path,
        needed_keys: set[str],
        source_prefix: str,
    ) -> dict[str, torch.Tensor]:
        with index_path.open() as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        shard_to_keys: dict[str, list[str]] = {}
        for key in needed_keys:
            if shard := weight_map.get(key):
                shard_to_keys.setdefault(shard, []).append(key)

        weights: dict[str, torch.Tensor] = {}
        for shard_filename, keys in shard_to_keys.items():
            shard_path = checkpoint_dir / shard_filename
            logger.debug(f"Reading {len(keys)} key(s) from shard {shard_filename}")
            with safe_open(str(shard_path), framework="pt") as f:
                for key in keys:
                    weights[self._remap_key(key, source_prefix)] = f.get_tensor(key)

        return weights

    def _build_config(
        self,
        source_config: dict,
        base_model: str,
        num_speculative_steps: int,
    ) -> MTPSpeculatorConfig:
        verifier_config_dict, _ = PretrainedConfig.get_config_dict(base_model)

        source_hidden = source_config.get("hidden_size")
        target_hidden = verifier_config_dict.get("hidden_size")
        if source_hidden and target_hidden and source_hidden != target_hidden:
            raise ValueError(
                f"Architecture mismatch: source MTP checkpoint has "
                f"hidden_size={source_hidden} but base_model '{base_model}' "
                f"has hidden_size={target_hidden}. Dimensions must match."
            )

        speculators_config = SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=num_speculative_steps)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=base_model,
                architectures=verifier_config_dict.get("architectures", []),
            ),
        )
        return MTPSpeculatorConfig(
            transformer_layer_config=source_config,  # type: ignore[arg-type]
            speculators_config=speculators_config,
        )

    def _save(
        self,
        config: MTPSpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        model = MTPDraftModel(config)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        self.validate_load_result(missing, unexpected)
        if missing:
            logger.debug(
                f"Keys not in extracted weights (loaded from verifier at runtime): "
                f"{missing}"
            )

        float_dtypes = {t.dtype for t in weights.values() if t.is_floating_point()}
        if float_dtypes:
            if len(float_dtypes) > 1:
                logger.warning(
                    f"Mixed float dtypes in checkpoint: {float_dtypes}. "
                    "Using first encountered dtype."
                )
            model.to(dtype=next(iter(float_dtypes)))  # type: ignore[call-arg]

        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        logger.info("Validating converted checkpoint...")
        try:
            MTPDraftModel.from_pretrained(str(output_path))
            logger.success("Validation succeeded")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
