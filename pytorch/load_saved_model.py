import torch
import torch.nn as nn

loaded_model = SimpleModel(input=20, middle=10, output=1)

loaded_model.load_state_dict(torch.load('file.pth'))

