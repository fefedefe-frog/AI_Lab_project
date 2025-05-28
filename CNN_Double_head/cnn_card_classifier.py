import os
import cv2
import torch
from torchvision import transforms
import torchmetrics
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CardDataset(Dataset):
    def __init__(self, csv_file_path: str, transform: transforms) -> None:
        self.csv_data = pd.read_csv(csv_file_path)
        self.transforms = transform

        self.seme_classes = sorted(self.csv_data["seme"].unique())
        self.numero_classes = sorted(self.csv_data["numero"].unique())

        self.seme_to_idx = {cls: idx for idx, cls in enumerate(self.seme_classes)}
        self.numero_to_idx = {cls: idx for idx, cls in enumerate(self.numero_classes)}

        self.idx_to_seme = {v: k for k, v in self.seme_to_idx.items()}
        self.idx_to_numero = {v: k for k, v in self.numero_to_idx.items()}

    def __len__(self) -> int:
        return len(self.csv_data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        riga = self.csv_data.iloc[idx]

        image_path = "../cnn_dataset_maker/" + str(riga["image_path"])
        seme = self.seme_to_idx[riga["seme"]]
        numero = self.numero_to_idx[riga["numero"]]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"{image_path} not found")

        if self.transforms:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out_img = self.transforms(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            out_img = torch.tensor(image, dtype=torch.float32)

        return out_img, torch.tensor(seme, dtype=torch.long), torch.tensor(numero, dtype=torch.long)


class DualHeadCNN(nn.Module):
    def __init__(self, num_classi_seme: int, num_classi_numeri: int):
        super(DualHeadCNN, self).__init__()

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
        )

        self.flatten = nn.Flatten()
        self.shared_layers = nn.Sequential(
            nn.Linear(256 * 30 * 22, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.testa_seme = nn.Linear(256, num_classi_seme)
        self.testa_numero = nn.Linear(256, num_classi_numeri)

    def forward(self, x):
        x_features = self.featuresExtractor(x)
        x_flatten = self.flatten(x_features)
        x_shared = self.shared_layers(x_flatten)

        logits_seme = self.testa_seme(x_shared)
        logits_numero = self.testa_numero(x_shared)

        return logits_seme, logits_numero


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset = CardDataset("../cnn_dataset_maker/output/dataset.csv", transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCNN(num_classi_seme=len(dataset.seme_classes), num_classi_numeri=len(dataset.numero_classes)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    metric_seme = torchmetrics.Accuracy(task='multiclass', num_classes=len(dataset.seme_classes)).to(device)
    metric_numero = torchmetrics.Accuracy(task='multiclass', num_classes=len(dataset.numero_classes)).to(device)

    train_losses_seme = []
    train_losses_numero = []
    testing_losses_seme = []
    testing_losses_numero = []
    accuracy_numero = []
    accuracy_seme = []
    acc_test_seme = []
    acc_test_numero = []

    def training_loop(dataloader):
        model.train()
        metric_seme.reset()
        metric_numero.reset()
        for batch, (images, seme_labels, numero_labels) in enumerate(dataloader):
            images, seme_labels, numero_labels = images.to(device), seme_labels.to(device), numero_labels.to(device)
            pred_seme, pred_numero = model(images)
            loss_seme = loss_fn(pred_seme, seme_labels)
            loss_numero = loss_fn(pred_numero, numero_labels)

            train_losses_seme.append(loss_seme.item())
            train_losses_numero.append(loss_numero.item())

            loss = loss_seme + loss_numero
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_seme.update(pred_seme, seme_labels)
            metric_numero.update(pred_numero, numero_labels)

        accuracy_seme.append(metric_seme.compute().item())
        accuracy_numero.append(metric_numero.compute().item())

    def testing_loop(dataloader):
        model.eval()
        metric_seme.reset()
        metric_numero.reset()
        with torch.no_grad():
            for images, seme_labels, numero_labels in dataloader:
                images, seme_labels, numero_labels = images.to(device), seme_labels.to(device), numero_labels.to(device)
                seme_logits, numero_logits = model(images)

                pred_seme = seme_logits.argmax(dim=1)
                pred_numero = numero_logits.argmax(dim=1)

                metric_seme.update(pred_seme, seme_labels)
                metric_numero.update(pred_numero, numero_labels)

                loss_s = loss_fn(seme_logits, seme_labels)
                loss_n = loss_fn(numero_logits, numero_labels)
                testing_losses_seme.append(loss_s.item())
                testing_losses_numero.append(loss_n.item())

        acc_test_seme.append(metric_seme.compute().item())
        acc_test_numero.append(metric_numero.compute().item())

    for epoch in range(50):
        print(f"Epoch {epoch+1}/50")
        training_loop(train_loader)
        testing_loop(test_loader)

    torch.save(model, "cnn_allenata.pth")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(train_losses_seme, label="Train Loss Seme", color='blue')
    axs[0, 0].plot(testing_losses_seme, label="Test Loss Seme", color='orange')
    axs[0, 0].set_title("Loss Seme")
    axs[0, 0].legend()

    axs[0, 1].plot(train_losses_numero, label="Train Loss Numero", color='green')
    axs[0, 1].plot(testing_losses_numero, label="Test Loss Numero", color='red')
    axs[0, 1].set_title("Loss Numero")
    axs[0, 1].legend()

    axs[1, 0].plot(accuracy_seme, label="Accuracy Seme", color='purple')
    axs[1, 0].plot(acc_test_seme, label="Test Accuracy Seme", color='pink')
    axs[1, 0].set_title("Accuracy Seme")
    axs[1, 0].legend()

    axs[1, 1].plot(accuracy_numero, label="Accuracy Numero", color='cyan')
    axs[1, 1].plot(acc_test_numero, label="Test Accuracy Numero", color='brown')
    axs[1, 1].set_title("Accuracy Numero")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results.png")
    plt.close()
