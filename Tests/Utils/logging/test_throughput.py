"""
Tests for ThroughputCallback.

The callback measures wall-clock duration between on_train_batch_start and
on_train_batch_end, then logs two metrics via pl_module.log:

  perf/time_per_step_ms  — elapsed time in milliseconds
  perf/samples_per_sec   — batch_size / elapsed_seconds

Design choices exercised here
------------------------------
- batch_size is read from trainer.train_dataloader.batch_size; the callback
  also unwraps a CombinedLoader by reading the `.loaders` attribute.
- If batch_size is None or elapsed <= 0 the callback silently returns without
  logging (guards against broken dataloaders or a monotonic-clock anomaly).

Test helpers
------------
_LogCapture       — records (name, value, kwargs) calls to pl_module.log.
_DataloaderStub   — minimal duck-type with a .batch_size attribute.
_CombinedLoader   — wraps a dataloader in a .loaders attribute (Lightning's
                    CombinedLoader shape).
_TrainerStub      — minimal trainer duck-type with .train_dataloader.
_make_module()    — returns a cast LightningModule whose .log is replaced by
                    a _LogCapture instance.
"""
import time
from typing import Any, Protocol, cast
from unittest.mock import patch

import lightning as L

from SkiNet.Utils.logging.throughput import ThroughputCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _HasLog(Protocol):
    def log(self, name: str, value: Any, **kwargs: Any) -> None:
        ...


class _LogCapture:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, dict]] = []

    def log(self, name: str, value: Any, **kwargs: Any) -> None:
        self.calls.append((name, value, kwargs))


class _DataloaderStub:
    def __init__(self, batch_size: int | None) -> None:
        self.batch_size = batch_size


class _CombinedLoader:
    """Mimics Lightning's CombinedLoader, which exposes its inner loader via .loaders."""

    def __init__(self, inner_batch_size: int) -> None:
        self.loaders = _DataloaderStub(inner_batch_size)


class _TrainerStub:
    def __init__(self, train_dataloader: Any) -> None:
        self.train_dataloader = train_dataloader


def _as_trainer(stub: _TrainerStub) -> L.Trainer:
    return cast(L.Trainer, stub)


def _make_module(capture: _LogCapture) -> L.LightningModule:
    return cast(L.LightningModule, capture)


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------

def test_logs_time_per_step_ms() -> None:
    """
    perf/time_per_step_ms must be logged as a positive float after a batch step.

    The test uses unittest.mock.patch to freeze perf_counter so elapsed is
    exactly 0.2 s → 200.0 ms, making the assertion deterministic.
    """
    callback = ThroughputCallback()
    capture = _LogCapture()
    module = _make_module(capture)
    trainer = _as_trainer(_TrainerStub(_DataloaderStub(batch_size=8)))

    with patch("time.perf_counter", side_effect=[0.0, 0.2]):
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

    names = [c[0] for c in capture.calls]
    assert "perf/time_per_step_ms" in names

    ms_value = next(c[1] for c in capture.calls if c[0] == "perf/time_per_step_ms")
    assert abs(ms_value - 200.0) < 1e-6


def test_logs_samples_per_sec() -> None:
    """
    perf/samples_per_sec must equal batch_size / elapsed_seconds.

    With batch_size=8 and elapsed=0.2 s → expected = 40.0 samples/s.
    """
    callback = ThroughputCallback()
    capture = _LogCapture()
    module = _make_module(capture)
    trainer = _as_trainer(_TrainerStub(_DataloaderStub(batch_size=8)))

    with patch("time.perf_counter", side_effect=[0.0, 0.2]):
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

    sps_value = next(c[1] for c in capture.calls if c[0] == "perf/samples_per_sec")
    assert abs(sps_value - 40.0) < 1e-6


def test_log_kwargs_on_step_true() -> None:
    """
    Both metrics must be logged with on_step=True, on_epoch=False, logger=True.

    These kwargs are required for the metrics to appear in step-level dashboards
    (TensorBoard, MLflow) rather than being averaged across epochs.
    """
    callback = ThroughputCallback()
    capture = _LogCapture()
    module = _make_module(capture)
    trainer = _as_trainer(_TrainerStub(_DataloaderStub(batch_size=4)))

    with patch("time.perf_counter", side_effect=[0.0, 0.1]):
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

    for _name, _value, kwargs in capture.calls:
        assert kwargs.get("on_step") is True
        assert kwargs.get("on_epoch") is False
        assert kwargs.get("logger") is True


# ---------------------------------------------------------------------------
# CombinedLoader unwrapping
# ---------------------------------------------------------------------------

def test_unwraps_combined_loader() -> None:
    """
    When train_dataloader is a CombinedLoader the callback must read batch_size
    from the inner loader exposed via .loaders, not from the CombinedLoader itself.

    Lightning wraps multiple dataloaders in a CombinedLoader; the outer object
    has no .batch_size attribute, so the callback uses getattr(dl, 'loaders', dl)
    to unwrap one level before reading .batch_size.
    """
    callback = ThroughputCallback()
    capture = _LogCapture()
    module = _make_module(capture)
    trainer = _as_trainer(_TrainerStub(_CombinedLoader(inner_batch_size=16)))

    with patch("time.perf_counter", side_effect=[0.0, 0.5]):
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

    sps_value = next(c[1] for c in capture.calls if c[0] == "perf/samples_per_sec")
    # 16 / 0.5 == 32.0
    assert abs(sps_value - 32.0) < 1e-6


# ---------------------------------------------------------------------------
# Guard conditions — no logging when inputs are invalid
# ---------------------------------------------------------------------------

def test_no_log_when_batch_size_is_none() -> None:
    """
    If batch_size cannot be determined (returns None) no metrics must be logged.

    This happens when a custom dataloader omits .batch_size, which is valid in
    PyTorch. Logging 'batch_size / elapsed' would raise ZeroDivisionError or
    produce nonsense, so the callback returns early.
    """
    callback = ThroughputCallback()
    capture = _LogCapture()
    module = _make_module(capture)
    trainer = _as_trainer(_TrainerStub(_DataloaderStub(batch_size=None)))

    with patch("time.perf_counter", side_effect=[0.0, 0.1]):
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

    assert capture.calls == []


def test_no_log_when_elapsed_is_zero() -> None:
    """
    If the two perf_counter reads return identical values (elapsed == 0) no
    metrics must be logged.

    A zero elapsed time would cause division by zero in samples_per_sec and
    would log a meaningless 0 ms step time, so the callback guards with
    `if elapsed <= 0: return`.
    """
    callback = ThroughputCallback()
    capture = _LogCapture()
    module = _make_module(capture)
    trainer = _as_trainer(_TrainerStub(_DataloaderStub(batch_size=8)))

    with patch("time.perf_counter", return_value=1.0):
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

    assert capture.calls == []


# ---------------------------------------------------------------------------
# Real-time sanity check (no mocking)
# ---------------------------------------------------------------------------

def test_real_elapsed_produces_positive_metrics() -> None:
    """
    Without time mocking, both metrics must be finite positive floats.

    This is a smoke test that exercises the full code path with a real
    perf_counter call, confirming the callback works end-to-end even on
    platforms where the clock resolution may be very high (sub-microsecond).
    """
    callback = ThroughputCallback()
    capture = _LogCapture()
    module = _make_module(capture)
    trainer = _as_trainer(_TrainerStub(_DataloaderStub(batch_size=32)))

    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    time.sleep(0.001)  # ensure non-zero elapsed
    callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

    assert len(capture.calls) == 2
    for _name, value, _kwargs in capture.calls:
        assert isinstance(value, float)
        assert value > 0.0
