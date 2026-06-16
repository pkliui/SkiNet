from typing import Any
import logging
import lightning as L
import math
import threading
import queue
import psutil
import torch

logger = logging.getLogger(__name__)


class SystemMetricsThreadCallback(L.Callback):
    """
    Callback that runs a background thread to periodically collect system metrics
    (CPU and RAM usage, GPU memory usage if available) and logs them to the trainer's loggers.
    Metrics are collected every `interval_sec` seconds and logged at the end of each training batch
    and validation epoch, as well as at the end of fitting.

    The callback ensures that the background thread is properly stopped when fitting ends.

    The callback is expected to be passed to the Lightning trainer's callbacks list,
    and it will automatically log system metrics to all of the trainer's loggers (including MLFlowLogger if it is enabled)
    without requiring any additional setup in the loggers themselves.

    Note that the native MLflow logging of system metrics is not supported by the MLFlowLogger itself.

    Note: all metric snapshots flushed at a given hook point are assigned the same global_step value,
    regardless of when they were actually collected within the interval. This means that if a validation
    epoch takes 30s and `interval_sec=5`, ~6 snapshots will all be logged at the same step, and the
    logger UI will not reflect the actual wall-clock time at which each snapshot was taken.

    Note: the collection cadence (`interval_sec`) and flush cadence (Lightning hooks) are decoupled, which
    has two opposite failure modes. If `interval_sec` is small relative to batch duration (e.g. `interval_sec=0.1`,
    batch=5s), ~50 snapshots accumulate before each flush and the queue may fill up, causing the oldest snapshots
    to be silently dropped — increase `max_queue_size` if you need to retain more history. Conversely, if
    `interval_sec` is large relative to batch duration (e.g. `interval_sec=5`, batch=0.1s), most batch-end
    flushes will find an empty queue and log nothing, as the thread has not collected a new snapshot yet.
    """

    def __init__(self, interval_sec: float = 5.0, max_queue_size: int = 256):
        self.interval_sec = float(interval_sec)
        self._max_queue_size = max_queue_size
        # acts as a sleeptimer and a signal to stop the thread when fitting ends
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # bounded queue to avoid memory growth
        self._metrics_queue: queue.Queue[dict[str, float]] = queue.Queue(maxsize=max_queue_size)

    def __getstate__(self) -> dict:
        # threading.Event and queue.Queue hold _thread.lock objects that cannot be pickled
        # (required by ddp_spawn which uses mp.spawn to transfer the callback to worker processes).
        # Exclude them; __setstate__ re-creates them so the callback works normally in the worker.
        state = self.__dict__.copy()
        state.pop("_stop_event", None)
        state.pop("_thread", None)
        state.pop("_metrics_queue", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._thread = None
        self._metrics_queue = queue.Queue(maxsize=self._max_queue_size)

    def _collect(self) -> dict[str, float]:
        """
        Create a dictionary of system metrics
        """
        vm = psutil.virtual_memory()
        m: dict[str, float] = {
            "system/cpu_percent": float(psutil.cpu_percent(interval=None)),
            "system/ram_percent": float(vm.percent),
            "system/ram_gb_used": float(vm.used / (1024**3)),
        }
        if torch.cuda.is_available():
            d = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(d)
            reserved = torch.cuda.memory_reserved(d)
            # tensors actively held by pytorch
            m["system/gpu_mem_allocated_gb"] = allocated / (1024**3)
            # pool pytorch claimed from OS (allocated + cached); gap between this and allocated is headroom before OOM
            m["system/gpu_mem_reserved_gb"] = reserved / (1024**3)
            try:
                gpu_util = torch.cuda.utilization(d)
                if gpu_util is not None:
                    m["system/gpu_util_percent"] = float(gpu_util)
            except ModuleNotFoundError:
                # nvidia-ml-py (pynvml) not installed — skip GPU utilisation metric
                pass
        return m

    @staticmethod
    def _sanitize_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """
        Remove values that logger backends cannot serialize, such as NaN and inf.
        """
        sanitized: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(float_value):
                sanitized[key] = float_value
        return sanitized

    def _run(self) -> None:
        """
        Runs in the background thread to periodically collect system metrics and put them in the queue for logging.

        If the queue is full, the oldest metric snapshot is dropped to make room for the new one,
        ensuring that the most recent metrics are always logged without unbounded memory growth.
        """
        # check the stop event every interval_sec seconds to know when to stop the thread (fitting ended)
        while not self._stop_event.is_set():
            try:
                # collect the metrics and put them in the queue for logging,
                metrics = self._collect()
                self._metrics_queue.put_nowait(metrics)
            # but don't block if the queue is full to avoid memory growth if the trainer is not consuming the metrics fast enough
            except queue.Full:
                try:
                    # drop the oldest metric and insert the new one
                    _ = self._metrics_queue.get_nowait()
                    self._metrics_queue.put_nowait(metrics)
                # if the main thread drained the queue via _flush_metrics between our get and put, the new metric is
                # already storable — but at this point put_nowait already succeeded above, so this only fires if
                # get_nowait itself raced; either way the snapshot is silently dropped rather than blocking.
                except queue.Empty:
                    pass
            # sleep for interval_sec seconds OR wake immediately if stop_event is set (fitting ended)
            self._stop_event.wait(self.interval_sec)

    def _flush_metrics(self, trainer: L.Trainer) -> None:
        """
        Read all currently queued metric snapshots from self._metrics_queue and log them to all of the Lightning trainer's
        loggers with the current global step. Stops when the queue is empty.

        Noe: global_step is a standard PyTorch Lightning Trainer attribute — it counts how many optimizer steps (batches)
        have been processed so far during training. It increments automatically as training progresses.
        """
        step = int(getattr(trainer, "global_step", 0))
        while True:
            try:
                metrics = self._metrics_queue.get_nowait()
            except queue.Empty:
                break
            metrics = self._sanitize_metrics(metrics)
            if not metrics:
                continue
            for lg in (trainer.loggers or []):
                try:
                    lg.log_metrics(metrics, step=step)
                except Exception:
                    logger.exception("Failed to log system metrics with logger %s", type(lg).__name__)
                    continue

    def _stop_thread(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        This is a Lightning hook method that is run once at the start of fitting to launch the background thread that collects system metrics.

        trainer and pl_module are passed in as arguments by Lightning when the hook is called, but they are not used in this method.
        """
        if self._thread is not None and self._thread.is_alive():
            return  # already running — don't spawn a second thread
        self._stop_event.clear()
        # drain any stale metrics left over from a previous fit that ended without a clean on_fit_end flush
        while not self._metrics_queue.empty():
            try:
                self._metrics_queue.get_nowait()
            except queue.Empty:
                break
        # prime CPU percent measurement once at the start of fitting as there is no previous data for the first measurement
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """
        This is a Lightning hook called at the end of each training batch,
        and it ensures that any collected system metrics are flushed to the loggers at the end of each training batch.
        """
        self._flush_metrics(trainer)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        This is a Lightning hook called at the end of each validation epoch,
        and it ensures that any collected system metrics are flushed to the loggers at the end of each validation epoch.
        """
        self._flush_metrics(trainer)

    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException) -> None:
        """
        This is a Lightning hook called when an exception occurs during training.
        It ensures the background thread is stopped even if on_fit_end is not called.
        """
        self._stop_thread()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        This is a Lightning hook method that is called by Lightning at the end of fitting,
        and it ensures that the background thread is properly stopped and all collected metrics are flushed to the loggers when fitting ends.
        """
        # signal the thread to stop and wait for it to finish, then flush any remaining metrics
        self._stop_thread()
        self._flush_metrics(trainer)
