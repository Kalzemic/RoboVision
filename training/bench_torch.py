# bench_pt.py
import numpy as np, time, torch
from model import RoboVision

device = 'cuda:0'
m = RoboVision(weights_file='RoboVision.pt').to(device).eval()

x = torch.rand(1, 3, 320, 320, device=device)

with torch.no_grad():
    for _ in range(50):                    # warm-up
        m(x)
    torch.cuda.synchronize()

    times = []
    t_start = time.perf_counter()
    for _ in range(1000):
        t0 = time.perf_counter()
        m(x)
        torch.cuda.synchronize()           # crucial
        times.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - t_start

import numpy as np
arr = np.array(times)
print(f"1000 iters in {elapsed:.2f}s  →  {1000/elapsed:.1f} FPS")
print(f"per-iter ms: mean {arr.mean():.2f}  p50 {np.percentile(arr,50):.2f}  "
      f"p95 {np.percentile(arr,95):.2f}  p99 {np.percentile(arr,99):.2f}")