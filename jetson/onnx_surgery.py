import onnx 
import onnx_graphsurgeon as gs
import numpy as np

graph = gs.import_onnx(onnx.load('RoboVision.onnx'))

boxes = [t for t in graph.outputs if t.name == 'boxes'][0]
scores = [t for t in graph.outputs if t.name == 'scores'][0]


num_detections = gs.Variable(
    name='num_detections',
    dtype=np.int32,
    shape=(1,1)
)

detection_boxes = gs.Variable(
    name='detection_boxes',
    dtype=np.float32,
    shape=(1,100,4)
)

detection_scores = gs.Variable(
    name='detection_scores',
    dtype=np.float32,
    shape=(1, 100),
)
detection_classes = gs.Variable(
    name='detection_classes',
    dtype=np.int32,
    shape=(1, 100),
)


nms_node = gs.Node(
    op='EfficientNMS_TRT',
    name='efficient_nms',
    attrs={
         'background_class': 0,          
        'max_output_boxes': 100,        
        'score_threshold': 0.5,         
        'iou_threshold': 0.5,
        'score_activation': False,     
        'box_coding': 0,               
        'plugin_version': '1',
    },
    inputs=[boxes, scores],
    outputs=[num_detections, detection_boxes, detection_scores, detection_classes],
)

graph.nodes.append(nms_node)

graph.outputs = [num_detections, detection_boxes, detection_scores, detection_classes]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), 'RoboVision_nms.onnx')

print("Saved RoboVision_nms.onnx")

new_graph = onnx.load('RoboVision_nms.onnx')
print(f"Inputs:  {[i.name for i in new_graph.graph.input]}")
print(f"Outputs: {[o.name for o in new_graph.graph.output]}")