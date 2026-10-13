# SPDX-License-Identifier: Apache-2.0
"""Agree a tensor-parallel store before any rank admits its chunks.

Each rank reserves on its own, copies, then reports here. A chunk is admitted
only when every rank is ready for it. Otherwise every rank drops the buffers
it staged for that chunk. A rank that never reports loses the round on
timeout, and the buffers already staged are dropped.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Callable
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# (request_id, start, end). One prefill can store the same request more than
# once, with a different token range each time.
GroupKey = tuple[str, int, int]
# (l1 owner id, object key). The key stays opaque so this module can be tested
# without constructing ObjectKey.
OwnedKey = tuple[int, Any]

CommitFn = Callable[[list[OwnedKey]], None]
AbortFn = Callable[[list[OwnedKey]], None]
PublishTokensFn = Callable[[list[int]], None]
StoreEndFn = Callable[[int, int], None]


@dataclass
class RankArrival:
    """One rank's result for a single store call."""

    worker_id: int
    world_size: int
    # Per chunk. True when this rank reserved the chunk, already had it, or
    # intentionally skipped it. False on allocation failure.
    ready: list[bool]
    keys_by_chunk: list[list[OwnedKey]]
    nbytes_by_chunk: list[int]
    chunk_size: int
    # False until the device copy into the staged buffers has finished.
    copy_done: bool = False
    publish_tokens: PublishTokensFn | None = None
    on_end: StoreEndFn | None = None


@dataclass
class _Group:
    world_size: int
    arrivals: dict[int, RankArrival] = field(default_factory=dict)
    timer: threading.Timer | None = None


@dataclass
class _Tombstone:
    """A round that already timed out. Late ranks still drop their buffers."""

    expires_at: float
    world_size: int
    seen: int
    waiting_copy: dict[int, RankArrival] = field(default_factory=dict)


class CrossRankStoreGate:
    """Collect one store report per rank, then commit or drop together."""

    def __init__(
        self,
        commit: CommitFn,
        abort: AbortFn,
        timeout_s: float = 5.0,
    ) -> None:
        """Create a gate.

        Args:
            commit: Admit these staged objects. Called with no lock held.
            abort: Drop these staged objects without admitting them.
            timeout_s: How long to wait for a missing rank.
        """
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive (got {timeout_s})")
        self._commit = commit
        self._abort = abort
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._groups: dict[GroupKey, _Group] = {}
        self._tombstones: dict[GroupKey, _Tombstone] = {}

    def submit(self, group_key: GroupKey, arrival: RankArrival) -> None:
        """Record one rank. Settle when every rank has reported and copied.

        Args:
            group_key: Request id plus the token range of this store.
            arrival: This rank's readiness and staged keys.
        """
        actions: list[Callable[[], None]] = []
        with self._lock:
            self._drop_expired_tombstones_locked()
            tomb = self._tombstones.get(group_key)
            if tomb is not None:
                actions.extend(self._late_locked(group_key, arrival, tomb))
            else:
                actions.extend(self._record_locked(group_key, arrival))
        for action in actions:
            action()

    def complete_copy(self, group_key: GroupKey, worker_id: int) -> None:
        """Mark that this rank's device copy has finished.

        Args:
            group_key: Request id plus the token range of this store.
            worker_id: Rank whose copy finished.
        """
        actions: list[Callable[[], None]] = []
        with self._lock:
            group = self._groups.get(group_key)
            arrival = group.arrivals.get(worker_id) if group is not None else None
            if arrival is not None and not arrival.copy_done:
                arrival.copy_done = True
                assert group is not None
                actions.extend(self._settle_if_ready_locked(group_key, group))
            else:
                tomb = self._tombstones.get(group_key)
                if tomb is None:
                    return
                late = tomb.waiting_copy.pop(worker_id, None)
                if late is None:
                    return
                actions.append(self._abort_action(late))
                self._note_seen_locked(group_key, tomb)
        for action in actions:
            action()

    def _record_locked(
        self, group_key: GroupKey, arrival: RankArrival
    ) -> list[Callable[[], None]]:
        group = self._groups.get(group_key)
        if group is None:
            group = _Group(world_size=max(arrival.world_size, 1))
            timer = threading.Timer(
                self._timeout_s, self._on_timeout, args=(group_key,)
            )
            timer.daemon = True
            group.timer = timer
            self._groups[group_key] = group
            timer.start()
        if arrival.worker_id in group.arrivals:
            logger.warning(
                "Cross-rank store ignored a second report from worker %s "
                "for request %s tokens [%s, %s)",
                arrival.worker_id,
                group_key[0],
                group_key[1],
                group_key[2],
            )
            return []
        group.arrivals[arrival.worker_id] = arrival
        return self._settle_if_ready_locked(group_key, group)

    def _settle_if_ready_locked(
        self, group_key: GroupKey, group: _Group
    ) -> list[Callable[[], None]]:
        if len(group.arrivals) < group.world_size:
            return []
        if not all(item.copy_done for item in group.arrivals.values()):
            return []
        if group.timer is not None:
            group.timer.cancel()
        self._groups.pop(group_key, None)
        return self._decide(group_key, list(group.arrivals.values()))

    def _decide(
        self, group_key: GroupKey, arrivals: list[RankArrival]
    ) -> list[Callable[[], None]]:
        width = max((len(item.ready) for item in arrivals), default=0)
        commit_mask = [
            all(index < len(item.ready) and item.ready[index] for item in arrivals)
            for index in range(width)
        ]
        request_id, start, end = group_key
        agreed = sum(commit_mask)
        if agreed:
            logger.info(
                "Cross-rank store committing request %s tokens [%s, %s): "
                "%d/%d chunks agreed",
                request_id,
                start,
                end,
                agreed,
                width,
            )
        else:
            logger.info(
                "Cross-rank store released request %s tokens [%s, %s): "
                "a rank was not ready",
                request_id,
                start,
                end,
            )
        actions: list[Callable[[], None]] = []
        committed_indices = [index for index, keep in enumerate(commit_mask) if keep]
        publisher = next(
            (item.publish_tokens for item in arrivals if item.publish_tokens is not None),
            None,
        )
        if publisher is not None and committed_indices:
            actions.append(lambda pub=publisher, idx=committed_indices: pub(idx))
        for item in arrivals:
            actions.extend(self._rank_actions(item, commit_mask))
        return actions

    def _rank_actions(
        self, arrival: RankArrival, commit_mask: list[bool]
    ) -> list[Callable[[], None]]:
        """Commit this rank's agreed chunks and drop the rest.

        Token bindings are published by the caller before these run, so a
        write-finished event never precedes the tokens it belongs to.
        """
        keep_pairs, drop_pairs, nbytes, n_chunks = _split(arrival, commit_mask)
        actions: list[Callable[[], None]] = []
        if drop_pairs:
            actions.append(lambda pairs=drop_pairs: self._abort(pairs))
        if keep_pairs:
            actions.append(lambda pairs=keep_pairs: self._commit(pairs))
        if arrival.on_end is not None:
            tokens = arrival.chunk_size * n_chunks if keep_pairs else 0
            reported_bytes = nbytes if keep_pairs else 0
            actions.append(
                lambda end=arrival.on_end, n=tokens, b=reported_bytes: end(n, b)
            )
        return actions

    def _late_locked(
        self, group_key: GroupKey, arrival: RankArrival, tomb: _Tombstone
    ) -> list[Callable[[], None]]:
        if arrival.copy_done:
            self._note_seen_locked(group_key, tomb)
            return [self._abort_action(arrival)]
        tomb.waiting_copy[arrival.worker_id] = arrival
        return []

    def _note_seen_locked(self, group_key: GroupKey, tomb: _Tombstone) -> None:
        tomb.seen += 1
        if tomb.seen >= tomb.world_size and not tomb.waiting_copy:
            self._tombstones.pop(group_key, None)

    def _on_timeout(self, group_key: GroupKey) -> None:
        actions: list[Callable[[], None]] = []
        with self._lock:
            group = self._groups.pop(group_key, None)
            if group is None:
                return
            if group.timer is not None:
                group.timer.cancel()
            request_id, start, end = group_key
            logger.warning(
                "Cross-rank store timed out for request %s tokens [%s, %s) "
                "with %d/%d ranks; releasing staged objects",
                request_id,
                start,
                end,
                len(group.arrivals),
                group.world_size,
            )
            tomb = _Tombstone(
                expires_at=time.monotonic() + 60.0,
                world_size=group.world_size,
                seen=0,
            )
            for arrival in group.arrivals.values():
                if arrival.copy_done:
                    tomb.seen += 1
                    actions.append(self._abort_action(arrival))
                else:
                    tomb.waiting_copy[arrival.worker_id] = arrival
            if tomb.seen < tomb.world_size or tomb.waiting_copy:
                self._tombstones[group_key] = tomb
        for action in actions:
            action()

    def _abort_action(self, arrival: RankArrival) -> Callable[[], None]:
        pairs = [pair for chunk in arrival.keys_by_chunk for pair in chunk]

        def _run(
            pairs: list[OwnedKey] = pairs,
            end: StoreEndFn | None = arrival.on_end,
        ) -> None:
            if pairs:
                self._abort(pairs)
            if end is not None:
                end(0, 0)

        return _run

    def _drop_expired_tombstones_locked(self) -> None:
        now = time.monotonic()
        expired = [
            key for key, tomb in self._tombstones.items() if tomb.expires_at <= now
        ]
        for key in expired:
            self._tombstones.pop(key, None)


def _split(
    arrival: RankArrival, commit_mask: list[bool]
) -> tuple[list[OwnedKey], list[OwnedKey], int, int]:
    keep: list[OwnedKey] = []
    drop: list[OwnedKey] = []
    nbytes = 0
    n_chunks = 0
    for index, chunk_keys in enumerate(arrival.keys_by_chunk):
        if (
            index < len(commit_mask)
            and commit_mask[index]
            and chunk_keys
        ):
            keep.extend(chunk_keys)
            n_chunks += 1
            if index < len(arrival.nbytes_by_chunk):
                nbytes += arrival.nbytes_by_chunk[index]
        else:
            drop.extend(chunk_keys)
    return keep, drop, nbytes, n_chunks
