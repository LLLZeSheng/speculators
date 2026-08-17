from scripts.patch_vllm_glm52_mtp_shared_weights import (
    PATCH_MARKER,
    build_patched,
)


SOURCE = '''def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        # Validate that weights were loaded for each expected MTP layer.
        loaded_layers: set[int] = set()
'''


def test_patch_routes_shared_weights_before_spec_layer_filter():
    patched = build_patched(SOURCE, "deepseek_mtp.py")

    shared_route = patched.index('if name == "model.embed_tokens.weight"')
    spec_filter = patched.index("get_spec_layer_idx_from_weight_name")
    assert shared_route < spec_filter
    assert 'elif name == "lm_head.weight"' in patched
    assert "shared_head.head.weight" in patched
    assert "missing_shared_params" in patched
    assert patched.count(PATCH_MARKER) == 2
