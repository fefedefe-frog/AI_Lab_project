import torch

model = torch.load("separate_layer_v2/cnn_allenata_5.pth")
torch.save(model.state_dict(), "weights.pth")
