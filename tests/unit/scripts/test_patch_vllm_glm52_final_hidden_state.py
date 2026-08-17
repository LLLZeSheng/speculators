from scripts.patch_vllm_glm52_final_hidden_state import (
    PATCH_MARKER,
    REPLACEMENT,
    apply,
    backup_path,
    restore,
)


SOURCE = """def forward(self):
        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
"""


def test_apply_is_idempotent_and_restore_round_trips(tmp_path):
    target = tmp_path / "deepseek_v2.py"
    target.write_text(SOURCE, encoding="utf-8")

    assert apply(target) == 0
    patched = target.read_text(encoding="utf-8")
    assert patched.count(PATCH_MARKER) == 1
    assert REPLACEMENT in patched
    assert backup_path(target).read_text(encoding="utf-8") == SOURCE

    assert apply(target) == 0
    assert target.read_text(encoding="utf-8") == patched

    assert restore(target) == 0
    assert target.read_text(encoding="utf-8") == SOURCE


def test_apply_rejects_unknown_source(tmp_path):
    target = tmp_path / "deepseek_v2.py"
    target.write_text("pass\n", encoding="utf-8")

    try:
        apply(target)
    except RuntimeError as error:
        assert "unsupported vLLM source" in str(error)
    else:
        raise AssertionError("unknown source should be rejected")
