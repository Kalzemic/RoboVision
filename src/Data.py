import torch, json, os

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image


class HandDataset(Dataset):
    def __init__(self, coco_json_path: str, images_dir_path: str, transforms=None):
        super().__init__()
        
        with open(coco_json_path,'r') as f:
            coco = json.load(f)
        
        self.images_dir = images_dir_path
        self.images = coco['images']
        self.transforms = transforms
        self.anns = defaultdict(list)

        for ann in coco['annotations']:
            self.anns[ann['image_id']].append(ann)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_info = self.images[index]
        img_path = os.path.join(self.images_dir,img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        anns = self.anns[img_info['id']]

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x,y,x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': tv_tensors.BoundingBoxes(
                torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                format='XYXY',
                canvas_size=(img_info['height'], img_info['width'])
            ),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_info['id']]),  
        }


        img = F.to_image(img)
        img = F.to_dtype(img, torch.float32, scale=True)


        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
            



def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
        

