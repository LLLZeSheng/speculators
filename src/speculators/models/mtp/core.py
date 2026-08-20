"""MTP speculator model implementation."""

import logging
from typing import Any, ClassVar

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig
from transformers.masking_utils import create_causal_mask

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin
from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.models.mtp.model_definitions import (
    mtp_model_classes,
    resolve_model_type,
)
from speculators.models.utils import conditional_torch_compile
from speculators.proposals.greedy import GreedyTokenProposalConfig

logger = logging.getLogger(__name__)

__all__ = ["MTPDraftModel", "compute_step_weights"]

_IGNORE_INDEX = -100


class _ChunkedVocabularyHead(nn.Linear):
    """Project all MTP steps through one FSDP module invocation.

    FSDP2 may reshard a module after its backward hook. Calling a sharded
    ``lm_head`` once per logits chunk therefore lets checkpoint recomputation
    observe a DTensor weight after earlier chunks have completed. Keeping every
    step and chunk inside this single module boundary gives FSDP one coherent
    forward/backward lifecycle while still avoiding full-sequence logits.
    """

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        targets: tuple[torch.Tensor, ...] | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if targets is None:
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError("ordinary vocabulary projection requires a tensor")
            return nn.functional.linear(hidden_states, self.weight, self.bias)

        if isinstance(hidden_states, torch.Tensor) or chunk_size is None:
            raise TypeError("chunked vocabulary projection requires tensor tuples")
        if len(hidden_states) != len(targets):
            raise ValueError("hidden-state and target step counts must match")

        step_losses: list[torch.Tensor] = []
        step_predictions: list[torch.Tensor] = []
        for step_hidden, step_targets in zip(hidden_states, targets, strict=True):
            loss_sum = torch.zeros(
                (), device=step_hidden.device, dtype=torch.float32
            )
            predictions: list[torch.Tensor] = []
            for chunk_start in range(0, step_hidden.shape[1], chunk_size):
                chunk_end = min(chunk_start + chunk_size, step_hidden.shape[1])
                chunk_logits = nn.functional.linear(
                    step_hidden[:, chunk_start:chunk_end], self.weight, self.bias
                )
                loss_sum = loss_sum + nn.functional.cross_entropy(
                    chunk_logits.permute(0, 2, 1),
                    step_targets[:, chunk_start:chunk_end],
                    ignore_index=_IGNORE_INDEX,
                    reduction="sum",
                ).float()
                predictions.append(chunk_logits.detach().argmax(dim=-1))
            step_losses.append(loss_sum)
            step_predictions.append(torch.cat(predictions, dim=1))

        return torch.stack(step_losses), torch.stack(step_predictions)


def compute_step_weights(beta: float = 0.6, num_steps: int = 3) -> list[float]:
    """Compute normalized exponential-decay step weights.

    alpha_k = beta^(k-1) / sum(beta^(j-1) for j=1..K)

    See FastMTP (arXiv:2509.18362), Equation 2.
    """
    raw = [beta**k for k in range(num_steps)]
    total = sum(raw)
    return [w / total for w in raw]


@SpeculatorModel.register("mtp")
class MTPDraftModel(DraftVocabMixin, SpeculatorModel):
    """MTP speculator model for multi-token prediction.

    Predicts multiple future tokens (default: 3) per forward pass using
    a single layer with weighted multi-step loss for training.

    embed_tokens and lm_head are managed by DraftVocabMixin — initialized
    to NaN, populated via load_verifier_weights() (called automatically by
    from_pretrained), and excluded from saved checkpoints.
    MTP does not create verifier_lm_head because it is not used in the forward
    pass.  Legacy converted checkpoints may still contain that key.
    """

    config_class: ClassVar[type[MTPSpeculatorConfig]] = MTPSpeculatorConfig  # type: ignore[misc]
    uses_verifier_lm_head: ClassVar[bool] = False
    lm_head_class: ClassVar[type[nn.Linear]] = _ChunkedVocabularyHead
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
    ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [  # type: ignore[misc]
        "verifier_lm_head.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)

        tc = self.config.transformer_layer_config
        if tc.model_type != "glm_moe_dsa":
            return

        # These two GLM implementation classes are private Transformers details
        # and have been renamed across releases.  Identify them by the raw
        # parameters that need special initialization instead of importing a
        # version-specific class name.  The checks are intentionally narrow so
        # ordinary Linear modules continue to use the parent initializer.
        if (
            isinstance(getattr(module, "weight", None), nn.Parameter)
            and isinstance(
                getattr(module, "e_score_correction_bias", None), nn.Parameter
            )
        ):
            nn.init.normal_(module.weight, mean=0.0, std=tc.initializer_range)
            nn.init.zeros_(module.e_score_correction_bias)
        elif (
            isinstance(getattr(module, "gate_up_proj", None), nn.Parameter)
            and isinstance(getattr(module, "down_proj", None), nn.Parameter)
        ):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=tc.initializer_range)
            nn.init.normal_(module.down_proj, mean=0.0, std=tc.initializer_range)

    def __init__(self, config: MTPSpeculatorConfig) -> None:
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "sdpa"
                if config.transformer_layer_config.model_type == "glm_moe_dsa"
                else "eager"
            )
        super().__init__(config=config)
        self._init_vocab(config)
        # MTP never consumes verifier_lm_head in forward.  DraftVocabMixin uses
        # uses_verifier_lm_head=False above to avoid allocating a third full-
        # vocabulary matrix (roughly 1.8 GiB per GLM 5.2 BF16 process).
        if self.use_draft_vocab:
            raise NotImplementedError(
                "Vocab reduction is not supported for MTP speculators"
            )

        tc = config.transformer_layer_config
        self._model_definitions = mtp_model_classes[resolve_model_type(tc.model_type)]
        self.mtp_layers = nn.ModuleList(
            [self._model_definitions.first_layer_class(tc, layer_idx=0)]
        )
        self.rotary_emb = self._model_definitions.rotary_emb_class(tc)

        self.post_init()

    @property
    def layers(self) -> nn.ModuleList:
        """Expose mtp_layers for FSDP wrapping compatibility."""
        return self.mtp_layers

    @property
    def target_layer_ids(self) -> list[int]:
        """MTP only uses the last hidden layer (verifier_last_hidden_states)."""
        return [self.config.transformer_layer_config.num_hidden_layers]

    def load_verifier_weights(self) -> None:
        """Re-set NaN sentinel before loading — meta-device init may clear
        it. Deletes verifier_lm_head after loading since MTP does not use it.
        """
        with torch.no_grad():
            self.embed_tokens.weight.fill_(torch.nan)
            self.lm_head.weight.fill_(torch.nan)
        super().load_verifier_weights()
        if self.config.transformer_layer_config.model_type == "glm_moe_dsa":
            for parameter in self.mtp_layers[0].self_attn.indexer.parameters():
                parameter.requires_grad_(False)

    # requires `dynamic=False`. See #876
    @conditional_torch_compile(dynamic=False)
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        step_weights: list[float] | None = None,
        logits_chunk_size: int | None = None,
        activation_checkpointing: bool = False,
        training_strategy: str = "full_unroll",
        sampled_step: int | None = None,
        return_logits: bool = True,
        return_dict: bool = True,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple:
        """Forward pass for MTP multi-token prediction (teacher-forced).

        At step k, uses ground-truth input_ids[t+k+1] as the embedding input and
        the MTP output from step k-1 (or verifier hidden states for step 0) as the
        hidden state input. Hidden states are passed recursively: each step's MTP
        output feeds the next step.

        Targets are derived from input_ids via per-step offset slicing -- no
        separate label tensor is needed. Use loss_mask to exclude positions
        (e.g. prompt tokens) from the loss.

        :param input_ids: Token IDs [1, seq_len]. Serves as both the
            embedding source and the prediction target (offset by step+2).
        :param hidden_states: Hidden states from verifier [1, seq_len, hidden_size]
        :param attention_mask: Optional attention mask [1, seq_len]
        :param position_ids: Optional position IDs [1, seq_len]
        :param loss_mask: Optional binary mask [1, seq_len]; 1=compute loss,
            0=ignore.
        :param step_weights: Per-step loss weights (None = uniform). Training only.
        :param logits_chunk_size: Split the sequence dimension before projecting to
            the full vocabulary. Each chunk is recomputed during backward, avoiding
            retention of an 8K-by-vocabulary logits tensor. Disabled when unset.
        :param activation_checkpointing: Recompute the MTP transformer layer during
            backward instead of retaining its intermediate activations.
        :param training_strategy: ``full_unroll`` retains the original recursive
            gradient graph. ``sampled_step`` builds a graph only for one selected
            prediction horizon and computes earlier recurrent states under no-grad.
        :param sampled_step: Prediction horizon selected by the trainer when using
            ``sampled_step``. All distributed ranks must use the same value.
        :param return_logits: Return per-step logits for inference/tests. Training can
            disable this to avoid retaining three full-vocabulary tensors.
        :param return_dict: Unused, kept for interface compatibility.
        :param kwargs: Absorbs unexpected batch keys
            (lengths, verifier_last_hidden_states)
        :return: Tuple of (logits_list, loss, metrics)
        """
        input_ids = input_ids.long()
        if logits_chunk_size is not None and logits_chunk_size <= 0:
            raise ValueError("logits_chunk_size must be positive when provided")
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        num_steps = self.config.num_speculative_steps

        if training_strategy not in {"full_unroll", "sampled_step"}:
            raise ValueError(f"unsupported MTP training strategy: {training_strategy}")
        if training_strategy == "sampled_step":
            if sampled_step is None or not 0 <= sampled_step < num_steps:
                raise ValueError(
                    f"sampled_step must be in [0, {num_steps}) for sampled_step "
                    f"training, got {sampled_step}"
                )
        elif sampled_step is not None:
            raise ValueError("sampled_step is only valid with sampled_step training")

        if step_weights is not None and len(step_weights) != num_steps:
            raise ValueError(
                f"step_weights has {len(step_weights)} entries but "
                f"num_speculative_steps={num_steps}; expected exactly "
                f"{num_steps} weights."
            )

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        all_logits: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float | torch.Tensor] = {}
        full_correct = torch.zeros((), dtype=torch.float32, device=device)
        full_total = torch.zeros((), dtype=torch.float32, device=device)
        accepted_draft_tokens = torch.zeros((), dtype=torch.float32, device=device)
        anchor_total = torch.zeros((), dtype=torch.float32, device=device)
        prefix_correct: torch.Tensor | None = None
        pending_chunked_steps: list[
            tuple[int, torch.Tensor, torch.Tensor, float, torch.Tensor]
        ] = []

        def record_step(
            step: int,
            loss_sum: torch.Tensor,
            predictions: torch.Tensor,
            step_targets: torch.Tensor,
            weight: float,
            valid_count: torch.Tensor,
        ) -> None:
            nonlocal total_loss, full_correct, full_total
            nonlocal accepted_draft_tokens, anchor_total, prefix_correct

            step_loss = weight * loss_sum / valid_count.clamp(min=1)
            total_loss = total_loss + step_loss
            metrics[f"loss_step_{step}"] = step_loss.detach().clone()

            with torch.no_grad():
                valid_targets = step_targets != _IGNORE_INDEX
                correct = predictions.eq(step_targets) & valid_targets
                correct_count = correct.float().sum()
                valid_count_float = valid_count.float()

                metrics[f"position_{step}_acc_sum"] = correct_count
                metrics[f"position_{step}_acc_total"] = valid_count_float
                full_correct = full_correct + correct_count
                full_total = full_total + valid_count_float

                # A sampled horizon deliberately does not project the no-grad
                # recurrent prefix, so prefix-acceptance metrics cannot be measured
                # honestly on this training step. Leave their denominator at zero;
                # full validation still reports the normal acceptance metrics.
                if training_strategy == "sampled_step":
                    return

                if prefix_correct is None:
                    conditional_total = valid_count_float
                    prefix_correct = correct
                    anchor_total = valid_count_float
                else:
                    conditional_total = (prefix_correct & valid_targets).float().sum()
                    prefix_correct = prefix_correct & correct

                prefix_correct_count = prefix_correct.float().sum()
                metrics[f"conditional_position_{step}_acc_sum"] = prefix_correct_count
                metrics[f"conditional_position_{step}_acc_total"] = conditional_total
                accepted_draft_tokens = accepted_draft_tokens + prefix_correct_count

        # Uniform valid_len keeps tensor shapes identical across loop
        # iterations, which torch.compile requires for stable codegen.
        # Cap steps so short sequences still produce partial results.
        effective_steps = min(num_steps, max(0, seq_len - 2))
        if (
            training_strategy == "sampled_step"
            and sampled_step is not None
            and sampled_step >= effective_steps
        ):
            raise ValueError(
                f"sampled_step={sampled_step} is unavailable for sequence length "
                f"{seq_len}; effective_steps={effective_steps}"
            )
        valid_len = seq_len - effective_steps - 1
        if valid_len <= 0 or effective_steps == 0:
            metrics["loss_sum"] = total_loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return (all_logits, total_loss, metrics)

        step_pos_ids = position_ids[:, :valid_len]
        causal_mask = create_causal_mask(
            config=self.config.transformer_layer_config,
            inputs_embeds=hidden_states[:, :valid_len],
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=step_pos_ids,
        )

        current_hidden = hidden_states
        for step in range(effective_steps):
            step_hidden = current_hidden[:, :valid_len]
            step_embeds = self.embed_tokens(
                input_ids[:, step + 1 : step + 1 + valid_len]
            )
            step_pos_emb = self.rotary_emb(step_hidden, step_pos_ids)

            def run_mtp_layer(
                layer_hidden: torch.Tensor,
                layer_embeds: torch.Tensor,
                layer_position_embeddings: Any = step_pos_emb,
            ) -> torch.Tensor:
                return self.mtp_layers[0](
                    hidden_states=layer_hidden,
                    token_embeddings=layer_embeds,
                    attention_mask=causal_mask,
                    position_ids=step_pos_ids,
                    position_embeddings=layer_position_embeddings,
                )

            selected_step = sampled_step is None or step == sampled_step
            # The sampled strategy intentionally avoids retaining autograd state for
            # the recurrent prefix. It preserves the exact hidden-state values used
            # by the selected horizon while bounding the graph to one GLM-MoE pass.
            grad_enabled = selected_step or training_strategy == "full_unroll"
            with torch.set_grad_enabled(torch.is_grad_enabled() and grad_enabled):
                checkpoint_layer = (
                    activation_checkpointing
                    and grad_enabled
                    and self.training
                    and torch.is_grad_enabled()
                )
                if checkpoint_layer:
                    mtp_output = checkpoint(
                        run_mtp_layer,
                        step_hidden,
                        step_embeds,
                        use_reentrant=False,
                    )
                else:
                    mtp_output = run_mtp_layer(step_hidden, step_embeds)

            if not selected_step:
                current_hidden = mtp_output.detach()
                continue

            step_targets = input_ids[:, step + 2 : step + 2 + valid_len]
            if loss_mask is not None:
                step_mask = loss_mask[:, step + 2 : step + 2 + valid_len]
                step_targets = step_targets.clone()
                step_targets[step_mask == 0] = _IGNORE_INDEX
            weight = step_weights[step] if step_weights is not None else 1.0
            if training_strategy == "sampled_step":
                # The trainer cycles horizons uniformly. Multiplying by K makes the
                # per-step loss an unbiased estimate of sum(alpha_k * loss_k).
                weight *= num_steps
            valid_count = (step_targets != _IGNORE_INDEX).sum()

            # The ordinary path remains the default for API compatibility. The
            # memory-efficient path projects only a bounded number of token
            # positions at once. Non-reentrant checkpointing discards each chunk's
            # full-vocabulary logits and recomputes it during backward.
            chunk_size = logits_chunk_size if not return_logits else None
            if chunk_size is None:
                logits = self.lm_head(mtp_output)
                if return_logits:
                    all_logits.append(logits)
                unreduced = nn.functional.cross_entropy(
                    logits.permute(0, 2, 1),
                    step_targets,
                    ignore_index=_IGNORE_INDEX,
                    reduction="none",
                )
                loss_sum = unreduced.float().sum()
                predictions = logits.detach().argmax(dim=-1)
                record_step(
                    step, loss_sum, predictions, step_targets, weight, valid_count
                )
            else:
                pending_chunked_steps.append(
                    (step, mtp_output, step_targets, weight, valid_count)
                )

            current_hidden = mtp_output

            if training_strategy == "sampled_step":
                break

        if pending_chunked_steps:
            chunk_size = logits_chunk_size
            if chunk_size is None:
                raise AssertionError("chunked steps require logits_chunk_size")
            step_hidden_states = tuple(item[1] for item in pending_chunked_steps)
            step_targets = tuple(item[2] for item in pending_chunked_steps)

            def project_all_steps(
                *hidden_steps: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                output = self.lm_head(
                    hidden_steps,
                    targets=step_targets,
                    chunk_size=chunk_size,
                )
                if not isinstance(output, tuple):
                    raise TypeError("chunked vocabulary head returned logits")
                return output

            if self.training and torch.is_grad_enabled():
                loss_sums, predictions = checkpoint(
                    project_all_steps,
                    *step_hidden_states,
                    use_reentrant=False,
                )
            else:
                loss_sums, predictions = project_all_steps(*step_hidden_states)

            for index, (step, _, targets, weight, valid_count) in enumerate(
                pending_chunked_steps
            ):
                record_step(
                    step,
                    loss_sums[index],
                    predictions[index],
                    targets,
                    weight,
                    valid_count,
                )

        metrics["full_acc_sum"] = full_correct
        metrics["full_acc_total"] = full_total
        metrics["accepted_draft_len_sum"] = accepted_draft_tokens
        metrics["accepted_draft_len_total"] = anchor_total.clone()
        metrics["accept_len_sum"] = accepted_draft_tokens + anchor_total
        metrics["accept_len_total"] = anchor_total.clone()
        metrics["loss_sum"] = total_loss.detach().clone()
        metrics["loss_total"] = torch.tensor(1.0, device=device)

        return (all_logits, total_loss, metrics)

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: PretrainedConfig,
        *,
        num_speculative_steps: int = 3,
        verifier_name_or_path: str | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> "MTPDraftModel":
        if verifier_name_or_path is None:
            raise ValueError(
                "verifier_name_or_path is required for MTP training. "
                "The verifier model must contain native MTP weights "
                "(mtp.* keys) to extract."
            )

        config = MTPSpeculatorConfig(
            transformer_layer_config=verifier_config,
            speculators_config=SpeculatorsConfig(
                algorithm="mtp",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=num_speculative_steps,
                    )
                ],
                default_proposal_method="greedy",
                # Read architectures from the verifier's published config.json (like
                # eagle3/dflash/peagle and the MTP converter). from_config would read
                # them off verifier_config, but that is the unwrapped text_config for
                # composite verifiers (e.g. Qwen3.5-MoE), which carries no
                # architectures -- leaving verifier.architectures empty.
                verifier=VerifierConfig.from_pretrained(verifier_name_or_path),
            ),
        )

        model = cls(config=config)

        from speculators.convert.mtp.converter import MTPConverter  # noqa: PLC0415

        converter = MTPConverter()
        state_dict = converter.convert_to_state_dict(
            verifier_name_or_path  # type: ignore[arg-type]
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        converter.validate_load_result(missing, unexpected)

        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for MTP.

        Step weights are computed from ``step_weight_beta`` and
        ``num_speculative_steps`` using the normalized exponential-decay
        formula from FastMTP (arXiv:2509.18362), Equation 2.

        Pass ``step_weights`` to override the computed weights.
        """
        step_weights = kwargs.get("step_weights")
        if step_weights is None:
            if "num_speculative_steps" not in kwargs:
                raise ValueError(
                    "num_speculative_steps must be set from the model config "
                    "before calling get_trainer_kwargs"
                )
            step_weights = compute_step_weights(
                beta=kwargs.get("step_weight_beta", 0.6),
                num_steps=kwargs["num_speculative_steps"],
            )
        train_kwargs: dict[str, Any] = {
            "step_weights": step_weights,
            "training_strategy": kwargs.get(
                "mtp_training_strategy", "full_unroll"
            ),
        }
        logits_chunk_size = kwargs.get("mtp_logits_chunk_size")
        memory_efficient = bool(logits_chunk_size)
        train_kwargs.update(
            logits_chunk_size=logits_chunk_size,
            activation_checkpointing=kwargs.get(
                "mtp_activation_checkpointing", False
            ),
            return_logits=not memory_efficient,
        )
        val_kwargs = train_kwargs.copy()
        val_kwargs["activation_checkpointing"] = False
        val_kwargs["training_strategy"] = "full_unroll"

        return train_kwargs, val_kwargs
