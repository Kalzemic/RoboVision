import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tidecv import TIDE, datasets
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from functools import partial
from src.Data import HandDataset, collate_fn

NUM_CLASSES = 2
BATCH_SIZE = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# --- Loade the Dataset ---

train_dataset = HandDataset('train.json', 'data/training_dataset/training_data/images')
val_dataset = HandDataset('val.json','data/validation_dataset/validation_data/images')

train_loader=  DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn)


# --- Load the trained model ---
model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
num_anchors = model.anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
model.head.classification_head = SSDLiteClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=NUM_CLASSES,
    norm_layer=norm_layer,
)
model.load_state_dict(torch.load('RoboVision.pt', map_location='cpu'))
model.to(device).eval()
# model.score_thresh = 0.3

# --- Run inference on val set, accumulate predictions ---
predictions = []
with torch.no_grad():
    for images, targets in tqdm(val_loader):
        images_dev = [img.to(device) for img in images]
        preds = model(images_dev)

        for pred, target in zip(preds, targets):
            image_id = target['image_id'].item()

            boxes = pred['boxes'].cpu()      # [N, 4] in xyxy
            scores = pred['scores'].cpu()    # [N]
            labels = pred['labels'].cpu()    # [N]

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],   # xyxy → xywh
                    "score": float(score.item()),
                })

# --- Save predictions ---
with open('predictions.json', 'w') as f:
    json.dump(predictions, f)

print(f"Saved {len(predictions)} predictions across {len(val_dataset)} images")

# --- Run TIDE ---
tide = TIDE()
tide.evaluate(
    datasets.COCO('val.json'),
    datasets.COCOResult('predictions.json'),
    mode=TIDE.BOX,
)

tide.summarize()      # prints error breakdown
tide.plot()           # renders bar charts of error contributions