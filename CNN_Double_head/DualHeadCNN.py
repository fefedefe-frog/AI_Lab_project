import torch
from torch import nn

class DualHeadCNN(nn.Module):
    def __init__(self, num_classi_seme: int= 4, num_classi_numeri: int= 13) -> None:
        super(DualHeadCNN, self).__init__()

        # Sezione della cnn che si occupa del feature extraction
        self.featuresExtractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Riduce l'output a (256, 4, 4)
        )

        # Sezione della cnn contenente le activation functions
        self.flatten = nn.Flatten()

        self.testa_seme = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classi_seme)
        )

        self.testa_numero = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classi_numeri)
        )


    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x_features = self.featuresExtractor(x)      # Estrazione delle features
        x_flatten = self.flatten(x_features)        # Trasformazione da array 3D a 2D

        logits_seme = self.testa_seme(x_flatten)     # Predizione del seme
        logits_numero = self.testa_numero(x_flatten) # Predizione del numero

        return logits_seme, logits_numero