import pytest
import torch
from torch import nn

from sglang.srt.models.qwen4_exp import Qwen4ExpPeerEmbedding


class FakePool:
    def __init__(self):
        self.published = False

    def ready(self, table):
        self.published = True

    def lookup_rows(self, table, ids):
        # The pool returns padded byte rows, not FP8 tensors.
        assert ids.shape == (2,)
        return torch.tensor([[56, 64, 99, 99], [64, 56, 99, 99]], dtype=torch.uint8)


def make_embedding():
    emb = Qwen4ExpPeerEmbedding.__new__(Qwen4ExpPeerEmbedding)
    nn.Module.__init__(emb)
    emb.pool = FakePool()
    emb.table = object()
    emb.row_start = 3
    emb.embedding_dim = 2
    emb.weight = nn.Parameter(
        torch.zeros((4, 2), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    emb.loaded_ranges = []
    return emb


def test_peer_ple_overlapping_checkpoint_shards_and_publish():
    emb = make_embedding()
    rows = torch.arange(16, dtype=torch.float32).reshape(8, 2).to(torch.float8_e4m3fn)
    emb.load_rows(rows[:5], 0, 5)
    with pytest.raises(ValueError, match="does not cover"):
        emb.publish()
    assert not emb.pool.published
    emb.load_rows(rows[5:], 5, 8)
    emb.publish()
    assert emb.pool.published
    torch.testing.assert_close(emb.weight.float(), rows[3:7].float())


def test_peer_ple_lookup_decodes_unpadded_fp8_bytes():
    emb = make_embedding()
    result = emb(torch.tensor([1, 2]))
    torch.testing.assert_close(
        result, torch.tensor([[1, 2], [2, 1]], dtype=torch.bfloat16)
    )
