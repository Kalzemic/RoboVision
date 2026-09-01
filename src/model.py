import torch 
import torch.nn as nn
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from functools import partial

NUM_CLASSES = 2

model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')


in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]

num_anchors = model.anchor_generator.num_anchors_per_location()

norm_layer = partial(nn.BatchNorm2d,eps=0.001,momentum=0.03)


model.head.classification_head = SSDLiteClassificationHead(in_channels=in_channels, num_anchors=num_anchors ,num_classes=NUM_CLASSES, norm_layer=norm_layer)

n_param = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters : {n_param/1e6:.2f}M')
print(f'Trainable parameters : {n_trainable/1e6:.2f}M')



images = [torch.randn(3,320,320), torch.randn(3,320,320)]

targets = [
    {
        'boxes':torch.tensor([[50., 60., 200., 240.],[220., 100., 300., 280.]]),
        'labels':torch.tensor([1,1],dtype=torch.int64)
    },
    {
        'boxes':torch.tensor([[100., 120., 260., 300.]]),
        'labels':torch.tensor([1],dtype=torch.int64)
    }
]

model.train() 

loss_dict = model(images,targets)

total_loss = sum(loss_dict.values())

print("\nTrain-mode losses:")
for k, v in loss_dict.items():
    print(f"  {k}: {v.item():.4f}")
print(f"  total: {total_loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = model(images)

print("\nEval-mode predictions:")
for i, p in enumerate(preds):
    print(f"  image {i}: boxes {tuple(p['boxes'].shape)}, "
          f"scores {tuple(p['scores'].shape)}, "
          f"labels {tuple(p['labels'].shape)}")