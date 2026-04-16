from typing import Any
import lightning as L
import threading
import queue
import psutil
import torch


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
    """

    def __init__(self, interval_sec: float = 5.0, max_queue_size: int = 256):
        self.interval_sec = float(interval_sec)
        # acts as a sleeptimer and a signal to stop the thread when fitting ends
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # bounded queue to avoid memory growth
        self._metrics_queue: queue.Queue[dict[str, float]] = queue.Queue(maxsize=max_queue_size)

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
            # pool pytorch clained from OS ie allocated + cached memory
            # gap between - headroom before OOM
            m["system/gpu_mem_reserved_gb"] = reserved / (1024**3)
            m["system/gpu_util_percent"] = float(torch.cuda.utilization(d))
        return m

    def _run(self) -> None:
        """
        Method that runs in the background thread to periodically collect system metrics and put them in the queue for logging.

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
                # in case the main thread has emptied the queue by calling _flush_metrics
                # as we reached e.g. end of batch or val epoch and the queue is now empty,
                except queue.Empty:
                    pass
            # sleep for interval_sec seconds OR wake immediately if stop_event is set (fitting ended)
            self._stop_event.wait(self.interval_sec)

    def _flush_metrics(self, trainer: L.Trainer) -> None:
        """
        Read all currently queued metric snapshots from self._metrics_queue and log them to all of the Lightning trainer's
        loggers with the current global step. Stops when the queue is empty.
        """
        step = int(getattr(trainer, "global_step", 0))
        while True:
            try:
                metrics = self._metrics_queue.get_nowait()
            except queue.Empty:
                break
            if not metrics:
                continue
            for lg in (trainer.loggers or []):
                lg.log_metrics(metrics, step=step)

                # try:
                #    lg.log_metrics(metrics, step=step)
                # except Exception:
                #    # Do not let one broken logger backend block system metrics for others.
                #    continue

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        This is a Lightning hook method that is run once at the start of fitting to launch the background thread that collects system metrics.

        trainer and pl_module are passed in as arguments by Lightning when the hook is called, but they are not used in this method.
        """
        if self._thread is not None and self._thread.is_alive():
            return  # already running — don't spawn a second thread
        self._stop_event.clear()
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

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        This is a Lightning hook method that is called by Lightning at the end of fitting,
        and it ensures that the background thread is properly stopped and all collected metrics are flushed to the loggers when fitting ends.
        """
        # signal the thread to stop and wait for it to finish, then flush any remaining metrics
        self._stop_event.set()
        if self._thread is not None:
            # wait for the thread to finish, but don't block indefinitely in case of issues with the thread
            self._thread.join(timeout=2.0)
        self._flush_metrics(trainer)
