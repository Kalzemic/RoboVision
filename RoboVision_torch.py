import cv2
import serial, struct
import torch
import numpy as np
import torch.nn as nn
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from functools import partial
from torchvision.ops import batched_nms
from model import RoboVision
NUM_CLASSES = 2
SCORE_THRESH = 0.5
IMAGE_FACTOR = 1

ORIG_H = 480
ORIG_W = 640
PORT = '/dev/ttyACM0'
BAUD = 115200
NUM_JOINTS = 2

def build_packet(joints):
    payload = struct.pack(f'<{NUM_JOINTS}f', *joints)
    return bytes([0xAA, 0x55]) + payload




def coords_to_angles(coords,image_dim):
    W,H = image_dim 
    x,y = coords 
    xc, yc = W / 2, H / 2
    x_norm, y_norm = (x - xc) / xc, (y - yc) / yc
    x_angle, y_angle = x_norm * (np.pi / IMAGE_FACTOR), y_norm * (np.pi / IMAGE_FACTOR)
    return -1 * x_angle, -1 * y_angle
    




def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
   # cv2.namedWindow("RoboVision", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    #cv2.resizeWindow("RoboVision", 800, 800)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = RoboVision()


    #model.load_state_dict(torch.load('RoboVision.pt', map_location='cpu'))
    model.to(device)
    model.eval()
    model.score_thresh = 0.7

    #ser = serial.Serial(PORT, BAUD)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        #print(frame.shape)
        frame_resized = cv2.resize(frame,(320,320))
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            boxes, scores = model(frame_t)

        
        boxes = boxes[0]
        hand_scores = scores[0,:,1]
        mask = hand_scores > SCORE_THRESH
        filtered_boxes = boxes[mask]
        filtered_scores = hand_scores[mask]

        max_score = 0
        max_box = None

        if len(filtered_boxes) > 0:
            labels = torch.zeros(len(filtered_boxes), dtype=torch.long,device=device)
            keep = batched_nms(filtered_boxes,filtered_scores, labels, iou_threshold=0.5)
            final_boxes = filtered_boxes[keep].cpu().numpy()
            final_scores = filtered_scores[keep].cpu().numpy()

            max_idx = final_scores.argmax()
            max_box = final_boxes[max_idx]
            max_score = final_scores[max_idx]
        if max_box is not None:
            x1, y1, x2, y2 = max_box.astype(int)
            scale_x = ORIG_W / 320
            scale_y = ORIG_H / 320
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            H, W = frame.shape[:2]
            x_angle, y_angle = coords_to_angles((cx,cy),(W,H))

            #ser.write(build_packet([float(x_angle), float(y_angle)]))

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


