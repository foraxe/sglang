# SPDX-License-Identifier: Apache-2.0
"""Startup rendezvous for independent engines sharing Engram row shards."""

import json
from datetime import timedelta
from functools import cache
from pathlib import Path

from torch.distributed import FileStore


class PeerGroup:
    def __init__(self, path: str, size: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.store = FileStore(path, size)
        self.store.set_timeout(timedelta(minutes=10))
        self.world_size = size
        self.rank_in_group = self.store.add("engram_rank", 1) - 1
        if self.rank_in_group >= size:
            raise RuntimeError("Engram peer group requires a fresh store per launch")
        self.step = 0

    def exchange(self, identity, value):
        self.step += 1
        keys = [f"engram/{self.step}/{rank}" for rank in range(self.world_size)]
        self.store.set(keys[self.rank_in_group], json.dumps((identity, value)))
        self.store.wait(keys)
        values = [json.loads(self.store.get(key)) for key in keys]
        if any(other != values[0][0] for other, _ in values):
            raise ValueError("Engram peers disagree on checkpoint or table geometry")
        return [value for _, value in values]


@cache
def join_peers(path: str, size: int) -> PeerGroup:
    return PeerGroup(path, size)
