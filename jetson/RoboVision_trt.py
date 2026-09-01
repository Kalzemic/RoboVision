import cv2 
import serial, struct  
from RoboVisionEngine import RoboVisionEngine
import numpy as np
import time
SCORE_THRESH = 0.3
IMAGE_FACTOR = 2

PORT = '/dev/ttyACM0'
BAUD = 115200
NUM_JOINTS = 2


def build_packet(joints):
    payload = struct.pack(f'<{NUM_JOINTS}f', *joints)
    return bytes([0xAA, 0x55]) + payload


def preprocess(frame_bgr):
    resized = cv2.resize(frame_bgr, (320, 320))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0     
    tensor = tensor.transpose(2, 0, 1)
    tensor = np.expand_dims(tensor, 0)
    return np.ascontiguousarray(tensor)

def coords_to_angles(coords, image_dim):
    W, H = image_dim
    x, y = coords
    xc, yc = W / 2, H / 2
    x_norm = (x - xc) / xc
    y_norm = (y - yc) / yc
    return -x_norm * (np.pi / IMAGE_FACTOR), -y_norm * (np.pi / IMAGE_FACTOR)


def main():
    engine = RoboVisionEngine('RoboVision.trt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    ser = serial.Serial(PORT, BAUD); time.sleep(2)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        orig_H, orig_W = frame.shape[:2]
        input_tensor = preprocess(frame)
        
        outputs = engine.infer(input_tensor)

        num_dets = int(outputs['num_detections'][0, 0])
        det_boxes = outputs['detection_boxes'][0]
        det_scores = outputs['detection_scores'][0]

        max_box = None 
        max_score = 0.0

        if num_dets > 0:
            valid_boxes = det_boxes[:num_dets]
            valid_scores =det_scores[:num_dets]

            idx = int(valid_scores.argmax())
            if valid_scores[idx] > SCORE_THRESH:
                max_box = valid_boxes[idx]
                max_score = valid_scores[idx]
            
            if max_box is not None:
                x1, y1, x2, y2 = max_box.astype(int)
                scale_x = orig_W / 320
                scale_y = orig_H / 320
                x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                H, W = frame.shape[:2]
                x_angle, y_angle = coords_to_angles((cx,cy),(W,H))

                ser.write(build_packet([float(x_angle), float(y_angle)]))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{x_angle}, {y_angle}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # else:
            #     ser.write(build_packet([float(-np.pi), float(-np.pi)]))
            # frame_display = cv2.resize(frame_resized, (800 , 800 ),
            #                    interpolation=cv2.INTER_NEAREST)
        cv2.imshow("RoboVision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



        

if __name__ == '__main__':
    main()