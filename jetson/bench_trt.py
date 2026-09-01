# bench_trt.py
import numpy as np, time
from RoboVisionEngine import RoboVisionEngine

engine = RoboVisionEngine('RoboVision.trt')
x = np.random.rand(1,3,320,320).astype(np.float32)

for _ in range(50): engine.infer(x)           # warm-up
times = []
t_start = time.perf_counter()
for _ in range(1000):
    t0 = time.perf_counter()
    engine.infer(x)
    times.append((time.perf_counter()-t0)*1000)
elapsed = time.perf_counter() - t_start
arr = np.array(times)
print(f"1000 iters in {elapsed:.2f}s  →  {1000/elapsed:.1f} FPS")
print(f"per-iter ms: mean {arr.mean():.2f}  p50 {np.percentile(arr,50):.2f}  "
      f"p95 {np.percentile(arr,95):.2f}  p99 {np.percentile(arr,99):.2f}")