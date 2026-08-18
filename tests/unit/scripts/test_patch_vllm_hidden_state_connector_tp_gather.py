from scripts.patch_vllm_hidden_state_connector_tp_gather import (
    IMPORT_ANCHOR,
    ORIGINAL,
    PATCH_MARKER,
    apply,
    backup_path,
    restore,
)


SOURCE = f'''import torch
{IMPORT_ANCHOR}
def save(kv_layer, req_slot_mapping_gpu, num_tokens):
    if True:
        with torch.cuda.stream(None):
{ORIGINAL}                return hidden_states_gpu
'''


def test_apply_is_idempotent_and_restore_round_trips(tmp_path):
    target = tmp_path / "example_hidden_states_connector.py"
    target.write_text(SOURCE, encoding="utf-8")

    assert apply(target) == 0
    patched = target.read_text(encoding="utf-8")
    assert patched.count(PATCH_MARKER) == 1
    assert "tensor_model_parallel_all_gather" in patched
    assert "hidden_states_gpu.shape[0] != num_tokens" in patched
    assert backup_path(target).read_text(encoding="utf-8") == SOURCE

    assert apply(target) == 0
    assert target.read_text(encoding="utf-8") == patched

    assert restore(target) == 0
    assert target.read_text(encoding="utf-8") == SOURCE


def test_apply_rejects_unknown_source(tmp_path):
    target = tmp_path / "example_hidden_states_connector.py"
    target.write_text("pass\n", encoding="utf-8")

    try:
        apply(target)
    except RuntimeError as error:
        assert "unsupported vLLM source" in str(error)
    else:
        raise AssertionError("unknown source should be rejected")
