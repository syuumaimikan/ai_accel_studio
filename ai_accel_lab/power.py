from __future__ import annotations
import threading, time


class NvmlPowerSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.01):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples = []
        self.available = False
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception as e:
            self.error = str(e)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        if not self.available: return
        self.samples.clear(); self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.available: return None
        self._stop.set()
        if self._thread: self._thread.join(timeout=1.0)
        return None if not self.samples else sum(self.samples)/len(self.samples)
