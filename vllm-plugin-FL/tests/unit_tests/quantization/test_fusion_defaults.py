import pytest

from vllm_fl.envs import fusion_enabled


@pytest.mark.parametrize("name", [
    "VLLM_FL_FA_ROPECAT", "VLLM_FL_D13_NORM_QUANT",
    "VLLM_FL_IR_NORM", "VLLM_FL_FLASH_KVCACHE_DIRECT_OUT",
])
def test_fusion_vendor_defaults_and_overrides(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VLLM_FL_FUSION_BUNDLE", raising=False)
    assert fusion_enabled(name, "metax")
    assert not fusion_enabled(name, "iluvatar")
    monkeypatch.setenv("VLLM_FL_FUSION_BUNDLE", "0")
    assert not fusion_enabled(name, "metax")
    monkeypatch.setenv(name, "1")
    assert fusion_enabled(name, "metax")
    assert not fusion_enabled(name, "iluvatar")
    monkeypatch.setenv("VLLM_FL_FUSION_BUNDLE", "1")
    monkeypatch.setenv(name, "0")
    assert not fusion_enabled(name, "metax")
