import json
import os
from PIL import Image, ImageDraw
import random


with open('../train.json') as f:
    coco = json.load(f)


random.seed(42)
for i, idx in enumerate(random.sample(range(len(coco['images'])), 4)):
    img_info = coco['images'][idx]
    img = Image.open(os.path.join('../data/training_dataset/training_data/images', img_info['file_name']))
    draw = ImageDraw.Draw(img)


    anns = [a for a in coco['annotations'] if a['image_id'] == img_info['id']]

    for a in anns:
        x, y, w, h = a['bbox']
        draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
        poly = a['segmentation'][0]
        draw.polygon(poly, outline='lime')

    img.save(f'check_{i+1}.jpg')