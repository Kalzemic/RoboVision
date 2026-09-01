import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.image_list import ImageList
from functools import partial


class RoboVision(nn.Module):
    def __init__(self, weights_file='RoboVision.pt',num_classes=2, input_size=(320, 320)):
        super().__init__()
        
        self.num_classes = num_classes

        self.model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
        in_channels = [module[0][0].in_channels for module in self.model.head.classification_head.module_list]

        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        

        norm_layer = partial(nn.BatchNorm2d,eps=0.001,momentum=0.03)

        self.model.head.classification_head = SSDLiteClassificationHead(in_channels=in_channels, num_anchors=num_anchors ,num_classes=self.num_classes, norm_layer=norm_layer)

        self.model.load_state_dict(torch.load(weights_file,map_location='cpu'))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        dummy = torch.zeros(1, 3, *input_size)
        image_list = ImageList(dummy,[input_size])

        with torch.no_grad():
            features = self.model.backbone(dummy)
            if isinstance(features,torch.Tensor):
                features = [features]
            else:
                features = list(features.values())
            anchors = self.model.anchor_generator(image_list,features)
        self.register_buffer('anchors',anchors[0])

    def decode_boxes(self, deltas, anchors):
        
        widths = anchors[:,2] - anchors[:,0]
        heights = anchors[:,3] - anchors[:,1]

        cx = anchors[:,0] + 0.5 * widths
        cy = anchors[:,1] + 0.5 * heights

        dx = deltas[..., 0] / 10.0
        dy = deltas[..., 1] / 10.0
        dw = deltas[..., 2] / 5.0
        dh = deltas[..., 3] / 5.0

        dw = torch.clamp(dw, max=4.135)
        dh = torch.clamp(dh, max=4.135)

        pred_cx = dx * widths + cx 
        pred_cy = dy * heights + cy
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        x1 = pred_cx - 0.5 * pred_w
        y1 = pred_cy - 0.5 * pred_h
        x2 = pred_cx + 0.5 * pred_w
        y2 = pred_cy + 0.5 * pred_h

        return torch.stack([x1, y1, x2, y2], dim=-1)  

    def forward(self,images):
        
        features = self.model.backbone(images)
        if isinstance(features,torch.Tensor):
                features = [features]
        else:
            features = list(features.values())
        
        head_outputs = self.model.head(features)

        bbox_regression = head_outputs['bbox_regression']
        cls_logits = head_outputs['cls_logits']

        boxes = self.decode_boxes(bbox_regression, self.anchors)

        scores = F.softmax(cls_logits,dim=-1)

        return boxes, scores

