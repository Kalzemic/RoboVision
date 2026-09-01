import torch 
from model import RoboVision



model = RoboVision()
model.eval()

dummy_input = torch.randn(1,3,320,320)

torch.onnx.export(model, dummy_input,'RoboVision.onnx',opset_version=18,input_names=['images'],output_names=['boxes','scores'],dynamic_axes=None,
                  do_constant_folding=True,verbose=False)

print('Exported to RoboVision.onnx')