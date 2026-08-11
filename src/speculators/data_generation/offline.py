import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_MIN_HIDDEN_STATES_NDIM = 2


def check_hidden_states(data: dict, tokens: list[int]) -> None:
    """Validate one hidden-state payload before it is cached or trained on."""
    required = {"token_ids", "hidden_states"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Hidden-state payload is missing keys: {sorted(missing)}")

    token_ids = data["token_ids"]
    hidden_states = data["hidden_states"]
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError("token_ids must be a torch.Tensor")
    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError("hidden_states must be a torch.Tensor")
    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must be 1-D, got shape {tuple(token_ids.shape)}")
    if hidden_states.ndim < _MIN_HIDDEN_STATES_NDIM:
        raise ValueError(
            f"hidden_states must have at least 2 dimensions, got "
            f"shape {tuple(hidden_states.shape)}"
        )
    if not hidden_states.is_floating_point():
        raise ValueError(
            f"hidden_states must be floating point, got {hidden_states.dtype}"
        )

    expected_token_ids = torch.as_tensor(
        tokens, dtype=token_ids.dtype, device=token_ids.device
    )
    if token_ids.shape != expected_token_ids.shape or not torch.equal(
        token_ids, expected_token_ids
    ):
        raise ValueError(
            "Token ids don't match the expected input "
            f"(actual shape={tuple(token_ids.shape)}, "
            f"expected shape={tuple(expected_token_ids.shape)})"
        )

    if len(tokens) != hidden_states.shape[0]:
        raise ValueError(
            f"Sequence length of hidden states {hidden_states.shape[0]}"
            f" doesn't match num tokens {len(tokens)}"
        )

    if not torch.isfinite(hidden_states).all().item():
        nan_count = int(torch.isnan(hidden_states).sum().item())
        posinf_count = int(torch.isposinf(hidden_states).sum().item())
        neginf_count = int(torch.isneginf(hidden_states).sum().item())
        raise ValueError(
            "Hidden states contain non-finite values: "
            f"NaN={nan_count}, +Inf={posinf_count}, -Inf={neginf_count}"
        )


def get_existing_hidden_state_indices(output_path: Path) -> list[int]:
    """Find existing `hs_i.safetensors` files (where i is the file index)"""

    existing_file_indices_set: set[int] = set()

    if not output_path.exists():
        return []

    for file_path in output_path.iterdir():
        if file_path.name.startswith("hs_") and file_path.name.endswith(".safetensors"):
            index_str = file_path.stem[3:]  # Remove "hs_" prefix
            try:
                file_index = int(index_str)
                existing_file_indices_set.add(file_index)
            except ValueError:
                continue

    return sorted(existing_file_indices_set)


def get_indices_to_process(
    num_samples: int,
    max_samples: int | None,
    existing: list[int],
    world_size: int,
    rank: int,
) -> list[int]:
    """Determines which indices should be processed. If max_samples is None
    returns all dataset indices not in existing. Otherwise gets the first
    `max_samples - len(existing)` samples not already in existing.

    Args:
        num_samples: Total size of preprocessed dataset
        max_samples: (Optional) limit for number of samples to process
        existing: list of ids that have already been processed
        world_size: Number of nodes to generate on
        rank: The rank of the local node

    Returns:
        list of dataset indices to process
    """

    target = min(max_samples, num_samples) if max_samples is not None else num_samples

    if target <= 0:
        return []

    chunk_size = target // world_size
    remainder = target % world_size
    # Distribute remainder across the first `remainder` ranks so chunks differ
    # by at most 1.
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)

    existing_s = set(existing)
    to_process = [i for i in range(start, end) if i not in existing_s]

    if not to_process:
        logger.info("All samples for this rank already processed!")
        return []

    if len(existing_s & set(range(start, end))) > 0:
        logger.info(
            f"Found {len(existing_s & set(range(start, end)))} existing samples"
            f" for rank {rank}."
        )

    return to_process
