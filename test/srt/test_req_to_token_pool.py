import pytest

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


@pytest.mark.parametrize("size", [1, 2, 8])
def test_req_to_token_pool_size_is_usable_capacity(size):
    pool = ReqToTokenPool(
        size=size,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
    )

    assert pool.available_size() == size

    slots = pool.alloc(size)
    assert slots == list(range(1, size + 1))
    assert pool.available_size() == 0
    assert pool.alloc(1) is None

    pool.free(slots)
    assert pool.available_size() == size

    pool.clear()
    assert pool.available_size() == size
