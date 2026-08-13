from unittest.mock import Mock

import scripts.train as train_module


def test_set_seed_does_not_call_cuda_seed_api(monkeypatch):
    manual_seed = Mock()
    cuda_seed = Mock(side_effect=AssertionError("CUDA seed API must not be called"))
    monkeypatch.setattr(train_module.torch, "manual_seed", manual_seed)
    monkeypatch.setattr(train_module.torch.cuda, "manual_seed_all", cuda_seed)

    train_module.set_seed(42)

    manual_seed.assert_called_once_with(42)
    cuda_seed.assert_not_called()


def test_empty_accelerator_cache_uses_current_accelerator(monkeypatch):
    empty_cache = Mock()
    monkeypatch.setattr(
        train_module.torch.accelerator,
        "current_accelerator",
        object,
    )
    monkeypatch.setattr(train_module.torch.accelerator, "empty_cache", empty_cache)
    cleanup = getattr(train_module, "empty_accelerator_cache", None)

    assert cleanup is not None
    cleanup()

    empty_cache.assert_called_once_with()


def test_empty_accelerator_cache_is_noop_without_accelerator(monkeypatch):
    empty_cache = Mock()
    monkeypatch.setattr(
        train_module.torch.accelerator,
        "current_accelerator",
        lambda: None,
    )
    monkeypatch.setattr(train_module.torch.accelerator, "empty_cache", empty_cache)
    cleanup = getattr(train_module, "empty_accelerator_cache", None)

    assert cleanup is not None
    cleanup()

    empty_cache.assert_not_called()
