# trt_parity.py
import cv2, numpy as np, onnxruntime as ort
from RoboVisionEngine import RoboVisionEngine

frame = cv2.imread('test.jpg')
resized = cv2.resize(frame, (320, 320))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
np_input = np.ascontiguousarray(
    (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
)

# ---- ONNX pre-NMS, then manual NMS in Python ----
sess = ort.InferenceSession('RoboVision.onnx', providers=['CPUExecutionProvider'])
boxes_ox, scores_ox = sess.run(['boxes', 'scores'], {'images': np_input})
hand_scores = scores_ox[0, :, 1]
mask = hand_scores > 0.5
kept_boxes = boxes_ox[0][mask]
kept_scores = hand_scores[mask]
print(f"ONNX: {mask.sum()} anchors above 0.5")
if len(kept_scores):
    order = np.argsort(-kept_scores)[:5]
    print("ONNX top-5 kept scores:", kept_scores[order].tolist())
    print("ONNX top-5 kept boxes:", kept_boxes[order].tolist())

# ---- TRT engine post-NMS ----
engine = RoboVisionEngine('RoboVision.trt')   # or RoboVision_fp32.trt
outputs = engine.infer(np_input)
n = int(outputs['num_detections'][0, 0])
print(f"\nTRT: num_detections = {n}")
print("TRT scores:", outputs['detection_scores'][0, :max(n,5)].tolist())
print("TRT boxes: ", outputs['detection_boxes'][0, :max(n,5)].tolist())
print("TRT classes:", outputs['detection_classes'][0, :max(n,5)].tolist())