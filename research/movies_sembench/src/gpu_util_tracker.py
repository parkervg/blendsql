import threading
import subprocess
import time
from contextlib import contextmanager


@contextmanager
def track_gpu(interval=0.01):
    """Track GPU utilization while code inside the context runs."""
    data = {"timestamps": [], "gpu_util": []}
    stop_event = threading.Event()

    def collect():
        start = time.time()
        while not stop_event.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            data["timestamps"].append(time.time() - start)
            data["gpu_util"].append(float(result.stdout.strip()))
            time.sleep(interval)

    thread = threading.Thread(target=collect)
    thread.start()
    try:
        yield data
    finally:
        stop_event.set()
        thread.join()
