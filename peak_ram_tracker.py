"""
Peak-RAM tracking helper.

psutil only gives you a *current* RSS snapshot -- there's no OS-level
"max RSS since I started watching" call that's portable and reliable across
platforms in the way torch.cuda.max_memory_allocated() is for GPU. So to get
a true peak, we poll RSS on a background thread at a short interval and keep
the running max. Overhead is small (a few memory reads per second).
"""
import threading
import time
import psutil
import os


class PeakRAMTracker:
    """Context manager that tracks peak RSS (in MB) for the current process
    while the `with` block runs.

    Usage:
        with PeakRAMTracker() as tracker:
            train_one_seed(...)
        peak_ram_mb = tracker.peak_mb
    """

    def __init__(self, poll_interval_s: float = 0.05):
        self.poll_interval_s = poll_interval_s
        self._process = psutil.Process(os.getpid())
        self._peak_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _poll_loop(self):
        while not self._stop_event.is_set():
            rss_mb = self._process.memory_info().rss / 1024**2
            if rss_mb > self._peak_mb:
                self._peak_mb = rss_mb
            self._stop_event.wait(self.poll_interval_s)

    def __enter__(self):
        # Take one immediate reading so short blocks still get a real number.
        self._peak_mb = self._process.memory_info().rss / 1024**2
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        return False  # don't suppress exceptions

    @property
    def peak_mb(self) -> float:
        return self._peak_mb