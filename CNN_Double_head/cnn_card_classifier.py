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
        self.csv_data = pd.read_csv(csv_file_path)  # Carico il csv usando pandas
        self.transforms = transform

        # Definisco le due classi di etichette
        self.seme_classes= sorted(self.csv_data["seme"].unique())
        self.numero_classes= sorted(self.csv_data["numero"].unique())

        # Converto le etichette testuali in numeri interni che la rete neurale può capire, usate ad esempio dalla loss_fn
        self.seme_to_idx = {cls: idx for idx, cls in enumerate(self.seme_classes)}
        self.numero_to_idx = {cls: idx for idx, cls in enumerate(self.numero_classes)}

        self.idx_to_seme = {v: k for k, v in self.seme_to_idx.items()}  # usate per quando voglio stampare le predizioni del modello in formato comprensibile per l'uomo
        self.idx_to_numero = {v: k for k, v in self.numero_to_idx.items()}

    def __len__(self) -> int:
        return len(self.csv_data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        riga= self.csv_data.iloc[idx]

        image_path= "..\cnn_dataset_maker\\" + str(riga["image_path"])
        seme= self.seme_to_idx[riga["seme"]]
        numero= self.numero_to_idx[riga["numero"]]

        # Leggo e converto l'immagine in un formato accettabile per la CNN
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"{image_path} not found")

        # Eseguo la conversione in tensor
        if self.transforms:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out_img = self.transforms(image)
        else:
            # fallback: converti a tensor senza trasformazioni
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            out_img = torch.tensor(image, dtype=torch.float32)

        return out_img, torch.tensor(seme, dtype=torch.long), torch.tensor(numero, dtype=torch.long)


# Una serie di trasformazioni da applicare alle immagini
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 180)), #TODO: trasformare da 80x60 a 128x128es stretcha l'immagine e quindi la distorce, meglio aggiungere dei bordi se si vuole l'immagine quadrata
    transforms.ToTensor(),  # converte e normalizza da [0–255] → [0–1]
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # ora [0–1] → [-1, 1]
])

dataset = CardDataset("../cnn_dataset_maker/output/dataset.csv", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Definizione rete neurale (CNN)
class DualHeadCNN(nn.Module):
    def __init__(self, num_classi_seme: int, num_classi_numeri: int):
        super(DualHeadCNN, self).__init__()

        # Sezione della cnn che si occupa di estrarre le feature dalle imamgini
        self.featuresExtractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),     # 240x180
            nn.ReLU(),
            nn.MaxPool2d(2),    # 120x90

            nn.Conv2d(32, 64, 3, padding=1),    # 120x90
            nn.ReLU(),
            nn.MaxPool2d(2),    # 60x45

            nn.Conv2d(64, 128, 3, padding=1),   # 60x45
            nn.ReLU(),
            nn.MaxPool2d(2),    # 30x22

            nn.Conv2d(128, 256, 3, padding=1),  # 30x22
            nn.ReLU(),
        )

        # Sezione con con activation function
        self.flatten = nn.Flatten()
        self.shared_layers= nn.Sequential(
            nn.Linear(256 * 30 * 22, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Due teste separate per la doppia classificazione
        self.testa_seme= nn.Linear(256, num_classi_seme)
        self.testa_numero= nn.Linear(256, num_classi_numeri)

    def forward(self, x):
        x_features = self.featuresExtractor(x)
        x_flatten= self.flatten(x_features)
        x_shared= self.shared_layers(x_flatten)

        logits_seme= self.testa_seme(x_shared)
        logits_numero= self.testa_numero(x_shared)

        return logits_seme, logits_numero


# Addestramento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualHeadCNN(num_classi_seme= 4, num_classi_numeri= 13).to(device)
print("Numero classi:", len(dataset.seme_to_idx), "+", len(dataset.numero_to_idx))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0004)

# Creiamo la matrice di accuratezza
metric_seme = torchmetrics.Accuracy(task='multiclass', num_classes=4).to(device)
metric_numero = torchmetrics.Accuracy(task='multiclass', num_classes=13).to(device)


train_losses_seme = []
train_losses_numero = []
testing_losses_seme = []
testing_losses_numero = []
accuracy_numero = []
accuracy_seme = []
acc_test_seme = []
acc_test_numero = []

def training_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    dataset_size = len(dataloader)

    # Reset delle metriche
    metric_seme.reset()
    metric_numero.reset()

    # Recupero il batch di dati dal disco
    for batch, (images, seme_labels, numero_labels) in enumerate(dataloader):

        # muovo i tensori di imaggini e label sul device corretto
        images, seme_labels, numero_labels = images.to(device), seme_labels.to(device), numero_labels.to(device)

        # calcolo le predizioni del modello
        pred_seme, pred_numero = model(images)

        # calcolo l'errore tra le predizioni del modello e i label corretti
        loss_seme = loss_fn(pred_seme, seme_labels)
        loss_numero = loss_fn(pred_numero, numero_labels)

        # Aggiorno array per plot
        train_losses_seme.append(loss_seme)
        train_losses_numero.append(loss_numero)

        # eseguo il passaggio di backpropagation
        loss= loss_seme + loss_numero
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # aggiornamento metriche
        metric_seme.update(pred_seme, seme_labels)
        metric_numero.update(pred_numero, numero_labels)

        # Stampa le statistiche ogni x batch(10 in questo caso)
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"\nLoss: {loss}, [{current:>5} / {dataset_size}]")

            acc_seme = metric_seme.compute()
            acc_numero = metric_numero.compute()
            print(f"Accuracy seme: {acc_seme} || {acc_seme * 100:.2f}%\nAccuracy numero: {acc_numero} || {acc_numero * 100:.2f}%")

    # stampo l'accuratezza alla fine del training
    acc_seme= metric_seme.compute()
    acc_numero = metric_numero.compute()

    # aggiorno gli array per plot
    accuracy_seme.append(acc_seme)
    accuracy_seme.append(acc_numero)
    print(f"Final Training Accuracy \n\t- Seme: {acc_seme} || {acc_seme * 100:.2f}%\n\t- Numero: {acc_numero} || {acc_numero * 100:.2f}%")



def testing_loop(dataloader, model):
    model.eval()

    metric_seme.reset()
    metric_numero.reset()

    # disabilito l'aggiornamento dei pesi
    with torch.no_grad():
        for images, seme_labels, numero_labels in dataloader:
            images, seme_labels, numero_labels = images.to(device), seme_labels.to(device), numero_labels.to(device)

            seme_logits, numero_logits = model(images)

            pred_seme= seme_logits.argmax(dim= 1)
            pred_numero= numero_logits.argmax(dim= 1)

            metric_seme.update(pred_seme, seme_labels)
            metric_numero.update(pred_numero, numero_labels)

            loss_n = loss_fn(numero_logits, numero_labels)
            loss_s = loss_fn(seme_logits, seme_labels)

            testing_losses_numero.append(loss_n)
            testing_losses_seme.append(loss_s)

    # stampo direttamente l'accuratezza finale
    acc_seme = metric_seme.compute()
    acc_numero = metric_numero.compute()

    # aggiorno gli array per plot
    acc_test_seme.append(acc_seme)
    acc_test_numero.append(acc_numero)
    print(f"\n\nFinal Testing Accuracy \n\t- Seme: {acc_seme} || {acc_seme * 100:.2f}%\n\t- Numero: {acc_numero} || {acc_numero * 100:.2f}%")

if __name__ == "__main__":
    epochs = 50
    # avvio l'allenamento del modello
    for e in range(epochs):
        print(f"\n\n====== Epoch {e}")
        training_loop(train_loader, model, loss_fn, optimizer)
        testing_loop(test_loader, model)
    print("Fatto!")

    torch.save(model, "cnn_allenata.pth")

    # Confusion Matrixs
    all_preds_seme = []
    all_labels_seme = []
    all_preds_numero = []
    all_labels_numero = []
    model.eval()
    with torch.no_grad():
        for images, seme_labels, numero_labels in test_loader:

            images = images.to(device)
            pred_seme, pred_numero = model(images)

            preds_seme = pred_seme.argmax(dim=1).cpu().numpy()
            all_preds_seme.extend(preds_seme)
            all_labels_seme.extend(seme_labels.numpy())

            preds_numero = pred_numero.argmax(dim=1).cpu().numpy()
            all_preds_numero.extend(preds_numero)
            all_labels_numero.extend(numero_labels.numpy())

    # Confusion Matrix per SEME
    cm_seme = confusion_matrix(all_labels_seme, all_preds_seme)
    disp_seme = ConfusionMatrixDisplay(confusion_matrix=cm_seme, display_labels=dataset.seme_classes)
    fig_seme, ax = plt.subplots()
    fig_seme.savefig("confusion_matrix_seme.png")
    disp_seme.plot(ax=ax)
    plt.title("Confusion Matrix - Seme")
    plt.show()

    # Confusion Matrix per NUMERO
    cm_numero = confusion_matrix(all_labels_numero, all_preds_numero)
    disp_numero = ConfusionMatrixDisplay(confusion_matrix=cm_numero, display_labels=dataset.numero_classes)
    fig_numero, ax = plt.subplots()
    fig_numero.savefig("confusion_matrix_numero.png")
    disp_numero.plot(ax=ax)
    plt.title("Confusion Matrix - Numero")
    plt.show()


    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Loss - Training e Testing
    axs[0, 0].plot(epochs, train_losses_seme, label="Train Loss", color='blue')
    axs[0, 0].plot(epochs, testing_losses_seme, label="Test Loss", color='orange')
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epoche")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Loss - Training e Testing
    axs[0, 0].plot(epochs, train_losses_numero, label="Train Loss", color='blue')
    axs[0, 0].plot(epochs, testing_losses_numero, label="Test Loss", color='orange')
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epoche")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Accuratezza numero
    axs[0, 1].plot(epochs, accuracy_numero, label="Training Accuracy Numero", color='green')
    axs[0, 1].set_title("Accuratezza Numero")
    axs[0, 1].set_xlabel("Epoche")
    axs[0, 1].set_ylabel("Accuratezza")
    axs[0, 1].grid(True)

    # Accuratezza seme
    axs[1, 0].plot(epochs, accuracy_seme, label="Training Accuracy Seme", color='purple')
    axs[1, 0].set_title("Accuratezza Seme")
    axs[1, 0].set_xlabel("Epoche")
    axs[1, 0].set_ylabel("Accuratezza")
    axs[1, 0].grid(True)

    # Accuratezza test numero
    axs[1, 0].plot(epochs, acc_test_seme, label="Testing Accuracy Seme", color='purple')
    axs[1, 0].set_title("Accuratezza Seme")
    axs[1, 0].set_xlabel("Epoche")
    axs[1, 0].set_ylabel("Accuratezza")
    axs[1, 0].grid(True)

    # Accuratezza test seme
    axs[1, 0].plot(epochs, acc_test_numero, label="Testing Accuracy Seme", color='purple')
    axs[1, 0].set_title("Accuratezza Seme")
    axs[1, 0].set_xlabel("Epoche")
    axs[1, 0].set_ylabel("Accuratezza")
    axs[1, 0].grid(True)

    # Spazio vuoto / futuro utilizzo (o puoi disegnare CM qui)
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("results.png")
    plt.close()
