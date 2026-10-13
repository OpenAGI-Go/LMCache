# SPDX-License-Identifier: Apache-2.0
"""Cross-rank store agreement: commit only chunks every rank has ready."""

# Standard
import threading

# First Party
from lmcache.v1.multiprocess.cross_rank_store import (
    CrossRankStoreGate,
    RankArrival,
)

_GROUP = ("req", 0, 256)


def _arrival(
    worker_id: int,
    ready: list[bool],
    keys: list[list[tuple[int, str]]] | None = None,
    copy_done: bool = True,
    world_size: int = 2,
) -> RankArrival:
    if keys is None:
        keys = [[] for _ in ready]
    return RankArrival(
        worker_id=worker_id,
        world_size=world_size,
        ready=ready,
        keys_by_chunk=keys,
        nbytes_by_chunk=[1] * len(ready),
        chunk_size=256,
        copy_done=copy_done,
    )


def test_both_ranks_ready_commits_and_publishes_once() -> None:
    committed: list[list[tuple[int, str]]] = []
    aborted: list[list[tuple[int, str]]] = []
    published: list[list[int]] = []
    ended: list[tuple[int, int, int]] = []
    gate = CrossRankStoreGate(
        commit=committed.append,
        abort=aborted.append,
        timeout_s=2.0,
    )
    first = _arrival(0, [True], keys=[[(1, "a")]])
    second = _arrival(1, [True], keys=[[(1, "b")]])
    first.publish_tokens = published.append
    first.on_end = lambda n, b: ended.append((0, n, b))
    second.on_end = lambda n, b: ended.append((1, n, b))

    gate.submit(_GROUP, first)
    assert committed == []
    gate.submit(_GROUP, second)

    assert aborted == []
    assert committed == [[(1, "a")], [(1, "b")]]
    assert published == [[0]]
    assert sorted(ended) == [(0, 256, 1), (1, 256, 1)]


def test_one_rank_not_ready_drops_the_other_ranks_buffers() -> None:
    committed: list[list[tuple[int, str]]] = []
    aborted: list[list[tuple[int, str]]] = []
    published: list[list[int]] = []
    gate = CrossRankStoreGate(
        commit=committed.append,
        abort=aborted.append,
        timeout_s=2.0,
    )
    ready = _arrival(0, [True], keys=[[(1, "a")]])
    ready.publish_tokens = published.append
    failed = _arrival(1, [False])
    gate.submit(_GROUP, ready)
    gate.submit(_GROUP, failed)

    assert committed == []
    assert aborted == [[(1, "a")]]
    assert published == []


def test_copy_must_finish_before_commit() -> None:
    committed: list[list[tuple[int, str]]] = []
    gate = CrossRankStoreGate(
        commit=committed.append,
        abort=lambda _pairs: None,
        timeout_s=2.0,
    )
    waiting = _arrival(0, [True], keys=[[(1, "a")]], copy_done=False)
    other = _arrival(1, [True], keys=[[(1, "b")]])
    gate.submit(_GROUP, waiting)
    gate.submit(_GROUP, other)
    assert committed == []
    gate.complete_copy(_GROUP, 0)
    assert committed == [[(1, "a")], [(1, "b")]]


def test_timeout_drops_the_rank_that_arrived() -> None:
    aborted = threading.Event()
    seen: list[list[tuple[int, str]]] = []

    def _abort(pairs: list[tuple[int, str]]) -> None:
        seen.append(pairs)
        aborted.set()

    gate = CrossRankStoreGate(commit=lambda _pairs: None, abort=_abort, timeout_s=0.05)
    gate.submit(_GROUP, _arrival(0, [True], keys=[[(1, "a")]]))
    assert aborted.wait(1.0)
    assert seen == [[(1, "a")]]


def test_already_resident_chunk_is_ready_without_a_new_buffer() -> None:
    """KEY_NOT_WRITABLE is reported by the caller as ready with no staged keys."""
    committed: list[list[tuple[int, str]]] = []
    aborted: list[list[tuple[int, str]]] = []
    gate = CrossRankStoreGate(
        commit=committed.append,
        abort=aborted.append,
        timeout_s=2.0,
    )
    # Worker 0 already has the chunk. Worker 1 just reserved it.
    present = _arrival(0, [True], keys=[[]])
    reserved = _arrival(1, [True], keys=[[(1, "b")]])
    gate.submit(_GROUP, present)
    gate.submit(_GROUP, reserved)
    assert aborted == []
    assert committed == [[(1, "b")]]
