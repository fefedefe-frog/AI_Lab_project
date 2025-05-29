import torch
from cnn_card_classifier import DualHeadCNN 

model = torch.load("output_6/cnn_allenata_2.pth")
torch.save(model.state_dict(), "weights.pth")
