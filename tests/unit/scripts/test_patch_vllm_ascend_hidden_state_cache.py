from scripts.patch_vllm_ascend_hidden_state_cache import (
    METHOD_ANCHOR,
    PATCH_MARKER,
    apply,
    restore,
)


SOURCE = f'''class CacheSpec:
{METHOD_ANCHOR}            "expected"
        )
        return specs[0]
'''


def test_apply_is_idempotent_and_restorable(tmp_path):
    target = tmp_path / "kv_cache_interface.py"
    target.write_text(SOURCE, encoding="utf-8")

    assert apply(target) == 0
    patched = target.read_text(encoding="utf-8")
    assert patched.count(PATCH_MARKER) == 1
    assert "Hidden-state and attention caches" in patched
    assert apply(target) == 0
    assert target.read_text(encoding="utf-8") == patched

    assert restore(target) == 0
    assert target.read_text(encoding="utf-8") == SOURCE
