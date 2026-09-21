# RoboVision

**A real-time perception-to-actuation loop running end-to-end on embedded hardware.**

A MobileNetV3 + SSDLite hand detector, trained in PyTorch, exported through ONNX (with graph surgery to splice in a fused NMS plugin), compiled to a TensorRT engine, and deployed on a Jetson Orin Nano. Detections are converted to servo angle commands and streamed over UART to an STM32F446 microcontroller, which drives PWM outputs to move physical servos toward the detected hand.

Camera pixel to motor motion, on-device, no cloud.

---

## System overview

```
   ┌─────────────┐   USB    ┌────────────────────────────────┐   UART   ┌──────────────────────┐
   │   Camera    │─────────▶│         Jetson Orin Nano       │─────────▶│      STM32F446       │
   └─────────────┘          │                                │  angles  │                      │
                            │  frame → preprocess → TensorRT │          │  angle → PWM CCR →   │
                            │  engine (SSDLite + Efficient   │          │  GPIO servo signal   │
                            │  NMS_TRT) → best-box → angle   │          │                      │
                            └────────────────────────────────┘          └──────────┬───────────┘
                                                                                   │ PWM
                                                                                   ▼
                                                                            ┌──────────────┐
                                                                            │    Servos    │
                                                                            └──────────────┘
```

Detected pixel coordinates are normalized to angles: the leftmost column of the frame maps to `−π/2`, the rightmost to `+π/2` (and analogously for vertical if a pan-tilt rig is used). The Jetson streams these angles over UART; the STM32 converts each incoming angle into a PWM compare-register value and updates the corresponding timer channel.

---

## Repo layout

```
RoboVision/
├── training/           # PyTorch: train MobileNetV3 + SSDLite on Hagrid
├── jetson/             # ONNX export, graph surgery, TensorRT build, on-device inference
├── stm/                # STM32F446 firmware (CMake-based) + host-side control scripts
└── README.md
```

Each stage has its own concerns and can be worked on independently. The contract between stages is deliberately small: `training/` produces a `.pt`, `jetson/` produces a `.trt` engine, `stm/` receives an angle vector on a serial port.

---

## Stage 1: Training (`training/`)

`Robovision.ipynb` is the canonical source of truth for training. It walks through data loading (Hagrid hand-gesture dataset, with Oxford data used for auxiliary experiments), model construction (MobileNetV3 backbone + SSDLite head from `torchvision.models.detection`), the training loop, evaluation, and export to `RoboVision.pt`.

Supporting scripts:

- `model.py`: model construction, shared between the notebook and inference code.
- `RoboVision_torch.py`: reference PyTorch inference for sanity-checking predictions and comparing against the TensorRT engine downstream.
- `bench_torch.py`: PyTorch-side latency benchmark, used as the baseline for the TensorRT speedup number.
- `parity.py`: numerical parity check. Same input through PyTorch and TensorRT, compares outputs to catch export bugs early.
- `tide.py`: TIDE error diagnostics on the detection outputs (false positive breakdown by type: localization, classification, background, etc.).

**Datasets, split manifests, sample images, and trained weights are not tracked in this repo**: point the notebook at your local Hagrid checkout and retrain to reproduce.

## Stage 2: Export and deployment (`jetson/`)

The pipeline from a `.pt` file to a running Jetson engine has three steps:

1. **ONNX export** (`onnx_export.py`). Wraps the SSDLite model to return raw pre-NMS outputs (box regression deltas + class scores per anchor), then calls `torch.onnx.export` with dynamic batch axes. Torchvision's SSD does NMS inside `forward()` in eval mode, so a wrapper is necessary to expose the raw tensors that the plugin will consume.

2. **Graph surgery** (`onnx_surgery.py`). Loads the exported ONNX with `onnx-graphsurgeon`, decodes the box deltas against the anchor grid inside the graph, then splices in an `EfficientNMS_TRT` node with the graph-level score threshold, IoU threshold, and max-detections attributes. The result is `RoboVision_nms.onnx`: a self-contained graph that produces final `(num_detections, boxes, scores, classes)` outputs.

3. **TensorRT engine build** (`trt_export.sh`).

   ```bash
   trtexec --onnx=RoboVision_nms.onnx --saveEngine=RoboVision.trt --fp16
   ```

   Must be run on the Jetson itself, not on a workstation. TensorRT engines are hardware-specific (compute capability, TRT version, JetPack version, precision mode all baked in). Rebuild the engine whenever any of those change.

Inference:

- `RoboVisionEngine.py`: thin wrapper around a deserialized TRT engine. Manages host/device buffers, execution context, and single-frame `.infer(image)` calls.
- `RoboVision_trt.py`: the on-device application loop. Pulls camera frames, runs inference, extracts the top-scoring box, converts to an angle vector, and streams it over UART to the STM32.
- `bench_trt.py`: end-to-end latency benchmark on the Jetson (loads the engine, runs `N` warmup + `N` timed inferences, reports min/mean/p95/max).

**Not tracked**: `.onnx` files, `.trt` engine, `.pt` weights. All are build artifacts or too large / device-specific to belong in git.

## Stage 3: Firmware (`stm/`)

STM32F446XX target, CMake-based build (no CubeIDE required, though the `.ioc` file is included so the project can be opened in CubeMX for peripheral inspection).

- `Core/`: application code (main loop, UART handler, PWM setup, ISRs).
- `Drivers/`: ST HAL (kept in-tree so the project builds without regenerating from the `.ioc`).
- `CMakeLists.txt`, `CMakePresets.json`, `cmake/`: build configuration.
- `startup_stm32f446xx.s`: startup assembly.
- `STM32F446XX_FLASH.ld`: linker script.
- `project_1.ioc`: CubeMX peripheral config (pins, clocks, timers, UART).
- `run.sh`: convenience script for build + flash.

The firmware receives an angle vector on a UART peripheral, parses it, and writes updated compare-register values into the timer channels responsible for servo PWM. Servo signal is standard 50 Hz PWM with 1–2 ms pulse width mapped to the requested angle.

**Host-side utilities** (in the same directory for co-location; run on a laptop, not on the STM32):

- `client.py`: sends test angle vectors over UART, useful for validating the STM32 side without needing the Jetson attached.
- `controller.py`: interactive keyboard control of the servos over UART.
- `test.py`: automated UART-side unit tests.

**Build:**

```bash
cd stm
cmake -S . -B build -G Ninja
cmake --build build
```

Flash with your preferred method (`st-flash`, `openocd`, or CubeProgrammer).

---

## Hardware

- **Compute:** NVIDIA Jetson Orin Nano (JetPack 6.x; TensorRT 10.x).
- **Microcontroller:** STM32F446 (Cortex-M4, 180 MHz).
- **Actuation:** Standard hobby servos on 50 Hz PWM.
- **Camera:** Any V4L2-compatible USB camera.
- **Interconnect:** UART between Jetson and STM32.

## Stack

- **Training:** PyTorch, torchvision (`ssdlite320_mobilenet_v3_large`).
- **Dataset:** Hagrid hand-gesture dataset (person / hand detection).
- **Export:** ONNX, `onnx-graphsurgeon`, `onnxsim`.
- **Deployment:** TensorRT (FP16), `EfficientNMS_TRT` plugin.
- **Firmware:** STM32 HAL, CMake, ARM GCC.
- **Interconnect:** UART framing over serial.

## Not in this repo

Deliberately excluded, must be produced or fetched locally:

- Datasets (Hagrid, Oxford) and their split manifests.
- Sample images and prediction dumps.
- Trained weights (`RoboVision.pt`).
- ONNX exports and TensorRT engines (device-specific; always rebuild on the target Jetson).
- STM32 CMake build directory (`stm/build/`).

---

## Notes on engineering choices

- **Why graph surgery instead of CPU-side NMS?** At Jetson-target frame rates, CPU-side NMS is a rounding error latency-wise but adds a GPU→CPU→GPU roundtrip and complicates the deployment story (Python post-processing on device). Fusing `EfficientNMS_TRT` into the graph gives a single-call inference interface with everything on-device.
- **Why UART, not USB or Ethernet, to the STM32?** UART is the simplest reliable framing for a small angle vector at low rate. No driver stack to fight, easy to bring up, easy to debug with a logic analyzer.
- **Why an STM32 at all, when the Jetson has GPIO?** Deterministic hard-real-time PWM is what an MCU is for. The Jetson runs Linux and can jitter; the STM32 timers run off hardware and give clean servo signals regardless of what the Jetson is doing.
- **Why FP16 on the engine?** Orin Nano has strong FP16 tensor cores, ~2× speedup over FP32 with negligible accuracy loss for this detector.
