"""
Tests for SystemMetricsThreadCallback.

Architecture under test
-----------------------
The callback uses a producer/consumer model to decouple metric collection from
the Lightning training loop:

  Background thread (_run)          Main thread (Lightning hooks)
  ──────────────────────────        ──────────────────────────────
  _collect() every interval_sec  →  queue.Queue  →  _flush_metrics()
                                                     called from hook callbacks

Concurrency design tested here
-------------------------------
- The queue is the only shared mutable state between the two threads.
  `queue.Queue` is thread-safe (uses an internal `threading.Lock`), so no
  additional locking is needed in the callback itself.

- The background thread uses `threading.Event.wait(interval_sec)` rather than
  `time.sleep()` so that `_stop_thread()` (which calls `_stop_event.set()`)
  can interrupt the sleep immediately, bounding shutdown latency to < 2 s.

- `put_nowait` / `get_nowait` (non-blocking queue ops) are used in the
  background thread to avoid ever blocking on the queue under back-pressure.
  If the queue is full the oldest snapshot is evicted to make room; this is
  safe because both operations are wrapped in try/except for the race where
  the main thread drains the queue between the two non-blocking calls.

- `on_fit_start` clears stale queue items *before* starting the thread, so
  the drain and thread-start are sequential and no lock is needed there.

- `on_fit_start` guards against double-spawning by checking `_thread.is_alive()`
  before creating a new thread; the id()-equality check in the test relies on
  the fact that Python does not reuse `id()` values for live objects.

Test helpers
------------
_RecordingLogger   — captures (metrics, step) pairs for assertion.
_FailingLogger     — always raises, used to verify per-logger error isolation.
_TrainerStub       — minimal duck-typed Lightning Trainer with `loggers` and
                     `global_step`.
_make_callback()   — creates a callback with a short interval (50 ms) so
                     thread-lifecycle tests complete quickly.
_null_trainer()    — trainer with no loggers; used when logging output is
                     irrelevant to the test.
"""
import logging
import math
import time
from typing import Any, cast

import lightning as L
import pytest

from SkiNet.Utils.logging.system_metrics import SystemMetricsThreadCallback


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], int]] = []

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        self.calls.append((metrics, step))


class _FailingLogger:
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        raise RuntimeError("logger backend failed")


class _TrainerStub:
    def __init__(self, loggers: Any, global_step: int = 7) -> None:
        self.loggers = loggers
        self.global_step = global_step


def _as_trainer(stub: _TrainerStub) -> L.Trainer:
    return cast(L.Trainer, stub)


def _as_module() -> L.LightningModule:
    return cast(L.LightningModule, None)


# ---------------------------------------------------------------------------
# _sanitize_metrics
# ---------------------------------------------------------------------------

def test_sanitize_metrics_drops_non_finite_values() -> None:
    """
    NaN, +inf, and -inf must be removed before passing metrics to logger backends.

    Some backends (e.g. MLflow) reject non-finite floats at the HTTP layer and
    raise exceptions. _sanitize_metrics is the single choke-point that filters
    them out so each logger does not have to guard individually.
    """
    metrics = {
        "ok": 1.5,
        "nan_value": math.nan,
        "pos_inf": math.inf,
        "neg_inf": -math.inf,
    }

    assert SystemMetricsThreadCallback._sanitize_metrics(metrics) == {"ok": 1.5}


def test_sanitize_metrics_drops_non_numeric_values() -> None:
    """
    Values that cannot be cast to float (e.g. strings) must be silently dropped.

    psutil or torch occasionally returns None or a sentinel string for metrics
    that are not available on the current platform; _sanitize_metrics must not
    propagate these to loggers.
    """
    metrics: dict[str, float] = {"ok": 2.0, "bad": "not-a-number"}  # type: ignore[dict-item]

    assert SystemMetricsThreadCallback._sanitize_metrics(metrics) == {"ok": 2.0}


def test_sanitize_metrics_empty_input_returns_empty() -> None:
    """An empty dict is a valid no-op input; the result must also be an empty dict."""
    assert SystemMetricsThreadCallback._sanitize_metrics({}) == {}


# ---------------------------------------------------------------------------
# _flush_metrics
# ---------------------------------------------------------------------------

def test_flush_metrics_empty_queue_is_noop() -> None:
    """
    _flush_metrics on an empty queue must not call any logger and must not raise.

    This is the common case when a hook fires but the background thread has not
    yet produced any snapshots (e.g. the very first on_train_batch_end after
    on_fit_start, before interval_sec has elapsed).
    """
    callback = SystemMetricsThreadCallback()
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=0)

    callback._flush_metrics(_as_trainer(trainer))

    assert logger.calls == []


def test_flush_metrics_uses_correct_step() -> None:
    """
    Each flushed snapshot must be tagged with trainer.global_step at flush time,
    not at collection time.

    All snapshots accumulated since the last flush are stamped with the *same*
    step value — the current training step — because the background thread has
    no access to the step counter. This is by design: it avoids a cross-thread
    read of a mutable integer and keeps the implementation simple.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/cpu_percent": 20.0})
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=42)

    callback._flush_metrics(_as_trainer(trainer))

    assert logger.calls == [({"system/cpu_percent": 20.0}, 42)]


def test_flush_metrics_filters_nan() -> None:
    """
    NaN values inside an otherwise valid snapshot must be stripped; the remaining
    finite values must still be logged.

    GPU utilization (system/gpu_util_percent) is NaN on some drivers when the
    device is idle, so mixed snapshots are common in practice.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/cpu_percent": 12.0, "system/gpu_util_percent": math.nan})
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=11)

    callback._flush_metrics(_as_trainer(trainer))

    assert logger.calls == [({"system/cpu_percent": 12.0}, 11)]


def test_flush_metrics_skips_all_nan_snapshot() -> None:
    """
    A snapshot where every value is non-finite must be dropped entirely rather
    than forwarded as an empty dict.

    Logging an empty metrics dict to MLflow or TensorBoard is a no-op at best
    and an API error at worst, so _flush_metrics short-circuits with `if not
    metrics: continue` after sanitization. The subsequent valid snapshot must
    still be processed normally.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/cpu_percent": math.nan})
    callback._metrics_queue.put({"system/ram_percent": 50.0})
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=1)

    callback._flush_metrics(_as_trainer(trainer))

    # the all-NaN snapshot is dropped; the second snapshot is logged
    assert logger.calls == [({"system/ram_percent": 50.0}, 1)]


def test_flush_metrics_continues_after_logger_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A RuntimeError raised by one logger backend must not prevent subsequent
    loggers from receiving the same snapshot.

    _flush_metrics iterates over trainer.loggers and wraps each call in
    try/except, so a broken backend (e.g. a misconfigured MLflow server)
    is isolated: it is logged as a warning and the loop moves on. This test
    places the failing logger first to confirm the good logger still fires.

    The failing logger emits an ERROR-level record via logger.exception(); we
    capture it with caplog so it is recorded rather than surfacing as bare
    "Captured stderr" noise in the test report, and assert the failure *was*
    logged — turning the expected error into a positive assertion.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/cpu_percent": 12.0})
    good_logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[_FailingLogger(), good_logger], global_step=11)

    with caplog.at_level(logging.ERROR, logger="SkiNet.Utils.logging.system_metrics"):
        callback._flush_metrics(_as_trainer(trainer))

    assert good_logger.calls == [({"system/cpu_percent": 12.0}, 11)]
    assert "Failed to log system metrics" in caplog.text


def test_flush_metrics_none_loggers_is_noop() -> None:
    """
    trainer.loggers may be None before the trainer is fully initialised.
    _flush_metrics must guard against this with `trainer.loggers or []` and
    must not raise AttributeError or TypeError.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/cpu_percent": 10.0})
    trainer = _TrainerStub(loggers=None, global_step=5)

    # must not raise
    callback._flush_metrics(_as_trainer(trainer))


def test_flush_metrics_drains_multiple_snapshots() -> None:
    """
    _flush_metrics must drain *all* accumulated snapshots in a single call, not
    just the first one.

    Multiple snapshots accumulate between hook invocations (e.g. several
    interval_sec cycles pass between two consecutive on_train_batch_end calls
    during a long batch). All of them must be forwarded in FIFO order, each
    tagged with the same current global_step.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/cpu_percent": 10.0})
    callback._metrics_queue.put({"system/cpu_percent": 20.0})
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=3)

    callback._flush_metrics(_as_trainer(trainer))

    assert len(logger.calls) == 2
    assert logger.calls[0] == ({"system/cpu_percent": 10.0}, 3)
    assert logger.calls[1] == ({"system/cpu_percent": 20.0}, 3)


# ---------------------------------------------------------------------------
# thread lifecycle
# ---------------------------------------------------------------------------

def _make_callback(interval_sec: float = 0.05) -> SystemMetricsThreadCallback:
    """Create a callback with a short collection interval for fast test execution."""
    return SystemMetricsThreadCallback(interval_sec=interval_sec)


def _null_trainer() -> _TrainerStub:
    """Trainer stub with no loggers; use when logging output is irrelevant."""
    return _TrainerStub(loggers=[], global_step=0)


def test_on_fit_start_spawns_thread() -> None:
    """
    on_fit_start must create and start the background collection thread.

    The finally block stops the thread unconditionally so the test never leaks
    a live daemon thread even on assertion failure — daemon=True means it would
    eventually be killed on interpreter exit, but explicit cleanup avoids
    interference with later tests in the same session.
    """
    callback = _make_callback()
    trainer = _null_trainer()

    callback.on_fit_start(_as_trainer(trainer), pl_module=_as_module())
    try:
        assert callback._thread is not None
        assert callback._thread.is_alive()
    finally:
        callback._stop_event.set()
        assert callback._thread is not None
        callback._thread.join(timeout=2.0)


def test_on_fit_start_does_not_spawn_second_thread() -> None:
    """
    Calling on_fit_start a second time while the thread is already alive must be
    a no-op — the existing thread object must not be replaced.

    Without this guard a Lightning Trainer that calls on_fit_start more than once
    (e.g. resume-from-checkpoint flows) would leak threads. The identity check
    uses `id()` of the live thread object; Python does not reuse id values for
    objects that are still referenced, so equality of ids means it is the same
    thread instance.
    """
    callback = _make_callback()
    trainer = _null_trainer()

    callback.on_fit_start(_as_trainer(trainer), pl_module=_as_module())
    first_thread_id = id(callback._thread)
    try:
        callback.on_fit_start(_as_trainer(trainer), pl_module=_as_module())
        assert id(callback._thread) == first_thread_id
    finally:
        callback._stop_event.set()
        assert callback._thread is not None
        callback._thread.join(timeout=2.0)


def test_on_fit_end_stops_thread_and_flushes() -> None:
    """
    on_fit_end must (1) signal the thread to stop via _stop_event, (2) join it
    with a timeout, and (3) flush any remaining snapshots to the loggers.

    The test sleeps 150 ms (3x the 50 ms interval) so the thread has time to
    produce at least one snapshot before on_fit_end is called.  After the call
    the thread must no longer be alive (join succeeded within the 2 s timeout)
    and the recording logger must have received at least one cpu_percent metric
    stamped with the correct global_step.

    Concurrency note: the 150 ms sleep creates a deliberate race window in which
    the background thread may enqueue 2-3 snapshots; the assertion `>= 1` is
    intentionally loose to remain stable across scheduling jitter.
    """
    callback = _make_callback()
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=1)

    callback.on_fit_start(_as_trainer(trainer), pl_module=_as_module())
    # let the thread collect at least one snapshot
    time.sleep(0.15)

    callback.on_fit_end(_as_trainer(trainer), pl_module=_as_module())

    assert callback._thread is not None
    assert not callback._thread.is_alive()
    # at least one metric snapshot must have been flushed
    assert len(logger.calls) >= 1
    # every flushed snapshot must contain cpu_percent and have valid values
    for metrics, step in logger.calls:
        assert "system/cpu_percent" in metrics
        assert step == 1


def test_on_exception_stops_thread() -> None:
    """
    on_exception must stop the background thread even when on_fit_end is never
    called (e.g. training aborts with an unhandled exception).

    Without this hook the daemon thread would keep sampling until the process
    exits, wasting CPU and potentially interfering with post-exception cleanup.
    The test calls join() explicitly (mirroring _stop_thread) so it does not
    rely on timing to detect a live thread.
    """
    callback = _make_callback()
    trainer = _null_trainer()

    callback.on_fit_start(_as_trainer(trainer), pl_module=_as_module())
    assert callback._thread is not None and callback._thread.is_alive()

    callback.on_exception(_as_trainer(trainer), pl_module=_as_module(), exception=RuntimeError("boom"))

    assert callback._thread is not None
    callback._thread.join(timeout=2.0)
    assert not callback._thread.is_alive()


# ---------------------------------------------------------------------------
# stale queue cleared on re-fit
# ---------------------------------------------------------------------------

def test_on_fit_start_drains_stale_queue_from_previous_fit() -> None:
    """
    on_fit_start must drain any metrics left over from a prior fit before
    launching the new background thread.

    Scenario: a fit completes but on_fit_end is not called (e.g. interrupted by
    KeyboardInterrupt before the hook fires), leaving stale snapshots in the
    queue. The next call to on_fit_start would otherwise mix those stale metrics
    into the new run's logs, potentially at the wrong global_step.

    The drain happens *before* the thread is started (sequential, no lock
    needed), so when the thread wakes up the queue contains only fresh data.

    Concurrency note: interval_sec=60 s is used so the thread's first collection
    cycle completes before it sleeps, but a second cycle cannot start within the
    0.1 s observation window. This avoids a race where the thread adds a new
    99.0-valued snapshot (coincidentally) before the assertions run.
    """
    # Use a long interval so the thread doesn't add a new snapshot during the assertion window.
    callback = SystemMetricsThreadCallback(interval_sec=60.0)
    # simulate leftover metrics from a prior fit that didn't flush cleanly
    callback._metrics_queue.put({"system/cpu_percent": 99.0})

    trainer = _null_trainer()
    callback.on_fit_start(_as_trainer(trainer), pl_module=_as_module())
    try:
        # The thread collects one snapshot immediately on entry, then sleeps for 60 s,
        # so the only item in the queue should be the fresh snapshot — not the stale 99.0 one!
        # Give the thread a moment to complete its first collection cycle.
        time.sleep(0.1)
        items = []
        while not callback._metrics_queue.empty():
            try:
                items.append(callback._metrics_queue.get_nowait())
            except Exception:
                break
        assert all(item.get("system/cpu_percent") != 99.0 for item in items), (
            "stale metric (99.0) should have been drained; found: %s" % items
        )
    finally:
        callback._stop_event.set()
        assert callback._thread is not None
        callback._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# hook delegation
# ---------------------------------------------------------------------------

def test_on_train_batch_end_flushes_metrics() -> None:
    """
    on_train_batch_end must delegate directly to _flush_metrics.

    This is a simple delegation test: a snapshot is pre-loaded into the queue
    (bypassing the background thread entirely - we do no call on_fit_start) and the hook is invoked on the
    main thread. The result must be identical to calling _flush_metrics directly.
    No thread is started, so there are no concurrency concerns here.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/cpu_percent": 5.0})
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=10)

    callback.on_train_batch_end(_as_trainer(trainer), pl_module=_as_module(), outputs=None, batch=None, batch_idx=0)

    assert logger.calls == [({"system/cpu_percent": 5.0}, 10)]


def test_on_validation_epoch_end_flushes_metrics() -> None:
    """
    on_validation_epoch_end must delegate directly to _flush_metrics.

    Same approach as test_on_train_batch_end_flushes_metrics: the queue is
    seeded from the main thread to avoid needing a live background thread and
    the associated timing sensitivity.
    """
    callback = SystemMetricsThreadCallback()
    callback._metrics_queue.put({"system/ram_percent": 42.0})
    logger = _RecordingLogger()
    trainer = _TrainerStub(loggers=[logger], global_step=20)

    callback.on_validation_epoch_end(_as_trainer(trainer), pl_module=_as_module())

    assert logger.calls == [({"system/ram_percent": 42.0}, 20)]


# ---------------------------------------------------------------------------
# queue-full drop-oldest behaviour
# ---------------------------------------------------------------------------

def test_run_drops_oldest_when_queue_is_full() -> None:
    """
    When the queue is at capacity the background thread must evict the oldest
    snapshot and insert the new one, keeping the queue bounded at max_queue_size.

    This tests the drop-oldest logic in _run, which uses two non-blocking queue
    operations (get_nowait + put_nowait) to avoid ever blocking the background
    thread:

        try:
            self._metrics_queue.put_nowait(metrics)   # fast path: queue has space
        except queue.Full:
            self._metrics_queue.get_nowait()           # evict oldest
            self._metrics_queue.put_nowait(metrics)    # insert newest

    Concurrency note: the second put_nowait can itself raise queue.Full if the
    main thread drained the queue between the get and the put (a benign TOCTOU
    race). The production code catches queue.Empty on the get to handle the
    symmetric race. Neither race can cause data corruption because queue.Queue
    uses an internal lock for every operation.

    This test pre-fills the 2-slot queue and gives the thread (interval=20 ms)
    six interval cycles (120 ms sleep) to attempt at least one drop-oldest
    cycle. After the thread is stopped the queue must still be ≤ 2 items and
    non-empty (confirming the thread actually ran).
    """
    callback = SystemMetricsThreadCallback(interval_sec=0.02, max_queue_size=2)
    # pre-fill the queue to capacity
    callback._metrics_queue.put({"system/cpu_percent": 1.0})
    callback._metrics_queue.put({"system/cpu_percent": 2.0})

    callback.on_fit_start(_as_trainer(_null_trainer()), pl_module=_as_module())
    # give the thread enough time to collect and attempt at least one drop-oldest cycle
    time.sleep(0.12)
    callback._stop_event.set()
    assert callback._thread is not None
    callback._thread.join(timeout=2.0)

    # the queue must still be bounded (not grown beyond max_queue_size)
    assert callback._metrics_queue.qsize() <= 2
    # and must be non-empty (thread did collect something)
    assert not callback._metrics_queue.empty()
