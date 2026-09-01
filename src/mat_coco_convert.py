import os
import json
import numpy as np
import scipy.io as sio
from PIL import Image


def convert(annotations_dir: str, images_dir: str, output_path: str):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "hand"}],
    }

    image_id = 0
    annotation_id = 0

    for file in sorted(os.listdir(annotations_dir)):
        if not file.endswith('.mat'):
            continue

        img_filename = file.replace('.mat', '.jpg')
        img_path = os.path.join(images_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"Skipping {file}: image {img_filename} not found")
            continue

        with Image.open(img_path) as img:
            W, H = img.size

        coco['images'].append({
            'id': image_id,
            'file_name': img_filename,
            'width': W,
            'height': H,
        })

        mat = sio.loadmat(os.path.join(annotations_dir, file),
                          squeeze_me=True, struct_as_record=False)
        boxes = mat['boxes']
        if not isinstance(boxes, np.ndarray):
            boxes = [boxes]

        for box in boxes:
            xs = [corner[1] for corner in [box.a, box.b, box.c, box.d]]  # [y, x] → x
            ys = [corner[0] for corner in [box.a, box.b, box.c, box.d]]  # [y, x] → y
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w, h = x_max - x_min, y_max - y_min

            segmentation = [c for corner in [box.a, box.b, box.c, box.d]
                              for c in (corner[1], corner[0])]

            coco['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [x_min, y_min, w, h],
                'area': w * h,
                'iscrowd': 0,
                'segmentation': [segmentation],
            })
            annotation_id += 1

        image_id += 1

    with open(output_path, 'w') as f:
        json.dump(coco, f)

    print(f"Wrote {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations to {output_path}")


if __name__ == '__main__':
    splits = [
        ('../data/training_dataset/training_data',   '../train.json'),
        ('../data/validation_dataset/validation_data', '../val.json'),
        ('../data/test_dataset/test_data',           '../test.json'),
    ]

    for split_dir, output in splits:
        ann_dir = os.path.join(split_dir, 'annotations')
        img_dir = os.path.join(split_dir, 'images')
        convert(ann_dir, img_dir, output)