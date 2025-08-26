from utils import SOHPredictor, ClassificationModel
import netron
import torch

pre = ClassificationModel(40, 9)
x = torch.randn(1, 40)
path = "./demo.pth"
torch.onnx.export(pre, x, path)
netron.start(path)