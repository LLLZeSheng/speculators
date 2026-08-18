from scripts.patch_vllm_ascend_extract_hidden_states_gather import (
    ORIGINAL,
    PATCH_MARKER,
    apply,
    backup_path,
    restore,
)


SOURCE = f'''def propose_draft_token_ids(self):
{ORIGINAL}                pass
'''


def test_apply_is_idempotent_and_restore_round_trips(tmp_path):
    target = tmp_path / "model_runner_v1.py"
    target.write_text(SOURCE, encoding="utf-8")

    assert apply(target) == 0
    patched = target.read_text(encoding="utf-8")
    assert patched.count(PATCH_MARKER) == 1
    assert "tensor_model_parallel_all_gather" in patched
    assert "state.shape[0] != num_scheduled_tokens" in patched
    assert backup_path(target).read_text(encoding="utf-8") == SOURCE

    assert apply(target) == 0
    assert target.read_text(encoding="utf-8") == patched

    assert restore(target) == 0
    assert target.read_text(encoding="utf-8") == SOURCE


def test_apply_rejects_unknown_source(tmp_path):
    target = tmp_path / "model_runner_v1.py"
    target.write_text("pass\n", encoding="utf-8")

    try:
        apply(target)
    except RuntimeError as error:
        assert "unsupported vLLM-Ascend source" in str(error)
    else:
        raise AssertionError("unknown source should be rejected")
