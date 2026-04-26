import torch

from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
    _flash_mla_with_optional_prefill_chunking,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def test_dsv4_flashmla_prefill_chunking_slices_row_tensors(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_FLASHMLA_PREFILL_CHUNK_SIZE", "4")

    calls = []

    def fake_runner(**kwargs):
        calls.append(
            {
                "q": kwargs["q"].clone(),
                "indices": kwargs["indices"].clone(),
                "topk_length": kwargs["topk_length"].clone(),
                "k_cache_shape": kwargs["k_cache"].shape,
                "backend": kwargs["backend"],
            }
        )
        return (kwargs["q"] + 1,)

    input_dict = {
        "q": torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1),
        "k_cache": torch.zeros(3, 128, 1, 1),
        "indices": torch.arange(10 * 64, dtype=torch.int32).view(10, 1, 64),
        "topk_length": torch.arange(10, dtype=torch.int32),
        "extra_indices_in_kvcache": None,
        "extra_topk_length": None,
    }

    out = _flash_mla_with_optional_prefill_chunking(
        input_dict=input_dict,
        backend="kernel",
        forward_mode=ForwardMode.EXTEND,
        runner=fake_runner,
    )

    assert out.shape == input_dict["q"].shape
    assert torch.equal(out, input_dict["q"] + 1)
    assert [call["q"].shape[0] for call in calls] == [4, 4, 2]
    assert [call["indices"].shape[0] for call in calls] == [4, 4, 2]
    assert [call["topk_length"].shape[0] for call in calls] == [4, 4, 2]
    assert all(call["k_cache_shape"] == input_dict["k_cache"].shape for call in calls)
    assert all(call["backend"] == "kernel" for call in calls)


def test_dsv4_flashmla_chunking_is_prefill_only(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_FLASHMLA_PREFILL_CHUNK_SIZE", "4")

    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs["q"].shape[0])
        return (kwargs["q"],)

    input_dict = {
        "q": torch.zeros(10, 1, 1, 1),
        "k_cache": torch.zeros(3, 128, 1, 1),
        "indices": torch.zeros(10, 1, 64, dtype=torch.int32),
        "topk_length": torch.ones(10, dtype=torch.int32),
        "extra_indices_in_kvcache": None,
        "extra_topk_length": None,
    }

    _flash_mla_with_optional_prefill_chunking(
        input_dict=input_dict,
        backend="kernel",
        forward_mode=ForwardMode.DECODE,
        runner=fake_runner,
    )

    assert calls == [10]
