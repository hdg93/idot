import torch
import torch.nn as nn
import torchvision

model_file = 'C:\Users\한동근\OneDrive - 한국폴리텍대학\바탕 화면\visual studio\python practice\yolov5\best.pt'
export_file = 'C:\Users\한동근\OneDrive - 한국폴리텍대학\바탕 화면\visual studio\python practice\yolov5\export.py'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_file)

dummy_input = torch.randn(1, 2, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    export_file,
    opset_version=11
)