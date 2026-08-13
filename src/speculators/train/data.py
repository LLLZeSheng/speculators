import json
import math
import os
import random
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
)
from speculators.train.long_context import (
    DISTANCE_STRETCH,
    SYNTHETIC_PREFIX,
    LongContextIndex,
    build_anchor_candidate_mask,
    concatenate_samples,
    stretched_position_ids,
)
from speculators.train.noise_transforms import TransformTensors

BatchType = dict[str, Any]


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            if not file.endswith("pt"):
                continue
            file_path = Path(root) / file
            datapath.append(file_path)

    return datapath


def split_files(datapath: str, ratio: float = 0.9, seed: int = 0):
    """Given a datapath, split the files into a training and validation set
    ratio is the proportion of files to put in the training set
    1 - ratio is the proportion of files to put in the validation set
    """
    random.seed(seed)
    file_list = list_files(datapath)
    random.shuffle(file_list)
    num_files = len(file_list)
    num_train_files = int(num_files * ratio)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]
    return train_files, val_files


# Data standardization functions
StandardizeFnSig = Callable[[dict[str, Any]], dict[str, Any]]


def create_empty_sample(
    hidden_size: int, num_target_layers: int = 3, dtype: torch.dtype = torch.bfloat16
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def standardize_data_v1(data: dict[str, Any]) -> dict[str, Any]:
    # v1 data format:
    # {
    #  "input_ids": [seq_len],
    #  "loss_mask": [seq_len],
    #  "hidden_states": [
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    ...
    #  ],
    # }

    return {
        "hidden_states": torch.cat(data["hidden_states"][:-1], dim=-1),
        "input_ids": data["input_ids"],
        "verifier_last_hidden_states": data["hidden_states"][-1],
        "loss_mask": data["loss_mask"],
    }


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


def _as_token_list(input_ids: Any) -> list[int]:
    """Normalize HF list/array/tensor token columns to plain Python integers."""
    values = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    return [int(token_id) for token_id in values]


def build_client_item(dataset_item: dict) -> ClientItem:
    """Build a request payload for vLLM hidden-state extraction.

    When ``messages`` is included, ``generate_hidden_states`` uses the Chat
    Completions API and vLLM **re-tokenizes from the raw messages**, ignoring
    ``input_ids``.  This is required for multimodal inputs (the Completions
    API cannot carry image/video/audio references), but harmful for text-only
    data: preprocessing truncates ``input_ids`` to ``seq_length``, yet the
    ``messages`` column stores the original un-truncated conversation.
    Re-tokenizing those messages produces a longer sequence that can exceed
    ``max_model_len``.

    We therefore only forward ``messages`` when the conversation actually
    contains multimodal content.  Text-only conversations always go through
    the Completions API with the pre-truncated ``input_ids``.

    This matters for models like Qwen3.5-0.8B whose ``AutoProcessor`` returns
    a ``ProcessorMixin`` (``Qwen3VLProcessor``), causing preprocessing to
    populate the ``messages`` column even for purely text-only datasets.
    Text-only EAGLE-3 models (e.g. Llama) use a plain tokenizer, so
    ``messages`` is never created and this guard is a no-op.
    """
    out_dict: dict = {"input_ids": _as_token_list(dataset_item["input_ids"])}

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]

    return cast("ClientItem", out_dict)


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
    ):
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def _compute_approx_lengths(self):
        raise NotImplementedError

    def _get_raw_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index) -> BatchType | None:
        data = self._get_raw_data(index)

        if data is None:
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Add lengths tensor
        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        # shape: [1]

        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        # shape: [seq_len]

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        #     "position_ids": [seq_len],
        # }

        # Apply transform
        if self.transform:
            data = self.transform(data)

        return data


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        transfer: HiddenStatesTransfer | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        train_ratio: float = 1.0,
        split: Literal["train", "val"] = "train",
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        validate_cached_hidden_states_finite: bool = True,
        long_context_index_path: str | None = None,
        long_context_block_size: int = 8,
        long_context_near_window: int = 8192,
        long_context_excluded_token_ids: tuple[int, ...] = (),
    ):
        self.data = load_from_disk(datapath)
        full_dataset_rows = len(self.data)
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be in (0.0, 1.0], got {train_ratio}")
        if split == "val" and train_ratio == 1.0:
            raise ValueError("train_ratio=1.0 leaves no validation split")

        # Both splits derive their boundary from this one expression,
        # so they are exactly complementary.
        split_idx = int(len(self.data) * train_ratio)
        start, stop = (
            (0, split_idx) if split == "train" else (split_idx, len(self.data))
        )
        if start >= stop:
            raise ValueError(
                f"{split} split is empty (dataset has {len(self.data)} rows, "
                f"train_ratio={train_ratio} gives split_idx={split_idx})"
            )
        self.start_file_idx = start
        self.data = self.data.select(range(start, stop))
        self._source_file_indices: list[int] | None = None
        if "source_index" in self.data.column_names:
            # Filtered HuggingFace datasets are re-numbered from zero while the
            # cached hidden-state files retain their original hs_<index> names.
            # Materialize the mapping once so every lookup is both correct and O(1).
            raw_source_indices = self.data.with_format(None)["source_index"]
            self._source_file_indices = [int(i) for i in raw_source_indices]
            if len(self._source_file_indices) != len(self.data):
                raise ValueError("source_index length doesn't match dataset length")
            if any(i < 0 for i in self._source_file_indices):
                raise ValueError("source_index values must be non-negative")

        self.transfer = transfer or FileTransfer(Path(datapath) / "hidden_states")
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        self.client: openai.OpenAI | None = None
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.validate_cached_hidden_states_finite = (
            validate_cached_hidden_states_finite
        )
        self.long_context_index = (
            LongContextIndex(long_context_index_path, full_dataset_rows)
            if long_context_index_path
            else None
        )
        self.long_context_block_size = long_context_block_size
        self.long_context_near_window = long_context_near_window
        self.long_context_excluded_token_ids = long_context_excluded_token_ids

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def _map_to_file_idx(self, index: int):
        if self._source_file_indices is not None:
            return self._source_file_indices[index]
        return index + self.start_file_idx

    def _setup_client(self):
        self.client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = self.client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id
        self.transfer.setup()

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        lengths = list(self.data.with_format(None)["seq_len"])
        if self.long_context_index is None:
            return lengths
        result = []
        for local_index, seq_len in enumerate(lengths):
            record = self.long_context_index[self.start_file_idx + local_index]
            result.append(
                min(self.max_len, int(seq_len) + record.prefix_len)
                if record.kind == SYNTHETIC_PREFIX
                else int(seq_len)
            )
        return result

    def __getitem__(self, index) -> BatchType | None:
        if self.long_context_index is None:
            return super().__getitem__(index)

        record = self.long_context_index[self.start_file_idx + index]
        target = self._get_raw_data(index)
        if target is None:
            return None
        target_len = target["input_ids"].shape[0]
        response_start = min(record.response_start, target_len)
        response_end = min(record.response_end, target_len)

        if record.kind == SYNTHETIC_PREFIX:
            synthetic = self._build_synthetic_prefix(target, target_len, record)
            if synthetic is None:
                return None
            data, target_offset = synthetic
            allowed_start = target_offset + response_start
            allowed_end = target_offset + response_end
            position_ids = torch.arange(data["input_ids"].shape[0], dtype=torch.long)
        else:
            data = target
            if record.kind == DISTANCE_STRETCH:
                if response_start < 0 or response_end <= response_start:
                    raise ValueError(
                        f"Invalid response boundary for stretched row {index}: "
                        f"[{response_start}, {response_end})"
                    )
                allowed_start = response_start
                allowed_end = response_end
                position_ids = stretched_position_ids(
                    target_len,
                    response_start,
                    record.target_anchor_position,
                    self.long_context_near_window,
                )
            else:
                # Preserve the original-data distribution. The rolling safety
                # check below still prevents crossing response/EOS boundaries.
                allowed_start = 0
                allowed_end = target_len
                position_ids = torch.arange(target_len, dtype=torch.long)

        data["lengths"] = torch.tensor([data["input_ids"].shape[0]], dtype=torch.long)
        data["position_ids"] = position_ids
        data["anchor_candidate_mask"] = build_anchor_candidate_mask(
            data["input_ids"],
            data["loss_mask"],
            self.long_context_block_size,
            self.long_context_excluded_token_ids,
            allowed_start,
            allowed_end,
        )
        if self.transform:
            data = self.transform(data)
        return data

    def _build_synthetic_prefix(self, target, target_len, record):
        donor_parts = []
        for donor_index in (record.donor1, record.donor2):
            donor = self._get_raw_data(donor_index - self.start_file_idx)
            if donor is None:
                return None
            donor_parts.append(donor)
        available = sum(part["input_ids"].shape[0] for part in donor_parts)
        prefix_len = min(record.prefix_len, available, self.max_len - target_len)
        # Keep the tail nearest B intact, then use the earlier donor for the
        # remaining prefix. Both donors are context-only and never supervised.
        take2 = min(prefix_len, donor_parts[1]["input_ids"].shape[0])
        take1 = prefix_len - take2
        prefix_parts = []
        for part, take in zip(donor_parts, (take1, take2), strict=True):
            if take:
                selected = {key: value[-take:] for key, value in part.items()}
                selected["loss_mask"] = torch.zeros_like(
                    selected["loss_mask"], dtype=torch.bool
                )
                prefix_parts.append(selected)
        target["loss_mask"] = target["loss_mask"].bool()
        return concatenate_samples([*prefix_parts, target]), prefix_len

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor] | None:
        if not self.client:
            self._setup_client()

        dataset_item = self.data[index]
        client_item = build_client_item(dataset_item)
        handle: str | None = None

        try:
            handle = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(handle)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            check_hidden_states(loaded_hs, _as_token_list(dataset_item["input_ids"]))

            file_idx = self._map_to_file_idx(index)
            match self.on_generate:
                case "cache":
                    self.transfer.cache(handle, file_idx)
                case "delete":
                    self.transfer.delete(handle)
        except Exception as e:  # noqa: BLE001 - sample boundary must not kill training
            if handle is not None:
                try:
                    self.transfer.delete(handle)
                except FileNotFoundError:
                    pass
                except Exception as cleanup_error:  # noqa: BLE001
                    warnings.warn(
                        f"Failed to remove invalid generated hidden states for "
                        f"sample {index}: {cleanup_error}",
                        stacklevel=1,
                    )
            warnings.warn(
                f"Invalid generated hidden states for sample {index}: {e}. Skipping...",
                stacklevel=1,
            )
            return None

        return loaded_hs

    def _get_raw_data(self, index):
        file_idx = self._map_to_file_idx(index)
        try:
            loaded_hs = self.transfer.get_cached(file_idx)
        except Exception as e:  # noqa: BLE001 - corrupt cache entry is skippable
            warnings.warn(
                f"Failed to read cached hidden states for sample {index} "
                f"(hs_{file_idx}.safetensors): {e}. Skipping...",
                stacklevel=1,
            )
            return None

        if loaded_hs is None:
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(index)
                case "skip":
                    return None
                case "warn":
                    warnings.warn(
                        f"Failed to load hidden states for sample {index}. Skipping...",
                        stacklevel=1,
                    )
                    return None
                case "raise":
                    raise RuntimeError(
                        f"Failed to load hidden states for sample {index}."
                    )

        if loaded_hs is None:
            return loaded_hs

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        expected_tokens = _as_token_list(self.data[index]["input_ids"])
        try:
            check_hidden_states(
                loaded_hs,
                expected_tokens,
                check_finite=self.validate_cached_hidden_states_finite,
            )
        except Exception as e:  # noqa: BLE001 - reject any invalid payload
            warnings.warn(
                f"Invalid cached hidden states for sample {index} "
                f"(hs_{file_idx}.safetensors): {e}. Skipping...",
                stacklevel=1,
            )
            return None

        return {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": self.data[index]["loss_mask"],  # [seq_len]
        }


class SampleFileDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform: TransformTensors | None = None,
        hidden_states_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the SampleFileDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory. All `.pt` files in this directory
            or its subdirectories will be loaded and used as training data. MUTUALLY
            EXCLUSIVE with `file_list`.
            file_list: The list of explict file paths to load data from. These files
            must be in the format produced by the Speculators generation scripts.
            MUTUALLY EXCLUSIVE with `datapath`.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
            standardize_fn: The function to standardize the data.

            Note: datapath or file_list must be provided, but not both.

        """

        if datapath is not None and file_list is not None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        if datapath is not None:
            file_list = list_files(datapath)

        if file_list is None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        self.data: list[str] = file_list

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples.

        First tries to load exact lengths from sample_lengths.json if available.
        Falls back to approximation based on file sizes.
        """
        # Look for the sample_lengths.json file
        sample_lengths_path = Path(self.data[0]).parent / "sample_lengths.json"
        if sample_lengths_path.exists():
            try:
                with sample_lengths_path.open() as f:
                    sample_lengths = json.load(f)
                # Extract file index from filename (e.g., data_42.pt -> 42)
                lengths = []
                for fname in self.data:
                    file_stem = Path(fname).stem
                    file_idx = file_stem.split("_")[-1]
                    lengths.append(sample_lengths[file_idx])
                return lengths
            except (KeyError, ValueError):
                pass

        # Fallback: approximate lengths from file sizes
        item_0 = self.__getitem__(0)
        if item_0 is None:
            raise ValueError(
                "Failed to load first element of datasets for length approximation"
            )
        lengths_0 = item_0["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def _get_raw_data(self, index):
        return standardize_data_v1(
            torch.load(
                self.data[index], mmap=True, weights_only=True, map_location="cpu"
            )
        )


def create_collate_fn(
    max_len: int,
    hidden_size: int,
    num_target_layers: int = 3,
    dtype: torch.dtype = torch.bfloat16,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        # Apply per-sample preprocessing and filter failed samples
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(hidden_size, num_target_layers, dtype=dtype)
            if preprocess:
                empty = preprocess(empty)
            batch = [empty]

        collated_data = {}
        for key in batch[0]:  # type: ignore[union-attr]
            if key == "lengths":
                collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
                continue
            # one copy per sample: preallocated buffer, hidden states cast during write
            first = batch[0][key]  # type: ignore[index]
            buffer_dtype = dtype if "hidden_states" in key else first.dtype
            out = torch.zeros(
                (max_len, *first.shape[1:]), dtype=buffer_dtype, device=first.device
            )
            offset = 0
            for b in batch:
                tensor = b[key]  # type: ignore[index]
                num_rows = min(tensor.shape[0], max_len - offset)
                out[offset : offset + num_rows] = tensor[:num_rows]
                offset += num_rows
                if offset == max_len:
                    break
            collated_data[key] = out.unsqueeze(0)
            # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data.pop("lengths")
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        lengths = torch.tensor(new_lengths, dtype=torch.long)

        # Create document_ids: maps each position to its document index, -1 for padding
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long), lengths
        )
        document_ids = torch.cat(
            [
                document_ids,
                -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
            ]
        ).unsqueeze(0)
        # shape: [1, max_len]
        collated_data["document_ids"] = document_ids

        return collated_data

    return collate_fn
