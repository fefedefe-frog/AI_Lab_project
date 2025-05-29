import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import cv2
from cnn_card_classifier import DualHeadCNN


# === CONFIGURAZIONE ===
MODEL_PATH = "cnn_allenata.pth"      # Path al tuo modello
IMAGE_PATH = "test_images/test3.png"        # Immagine da classificare
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Trasformazioni immagine (modifica se servono altre) ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 180)),
    transforms.ToTensor(),  # converte e normalizza da [0–255] → [0–1]
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # ora [0–1] → [-1, 1]
])

# === Carica immagine ===
if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"{IMAGE_PATH} non trovato")

img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_tensor = transform(img).unsqueeze(0).to(DEVICE)  # shape: (1, C, H, W)

# === Carica modello ===
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.eval()


# Definisco le classi per seme e numeri definiti come in CardDataset
classi_seme: tuple= ("cuori", "quadri", "fiori", "picche")
classi_numero: tuple= ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "K", "Q", "J")

# === Predizione ===
with torch.no_grad():
    output = model(input_tensor)

# === Interpreta output
# Caso 1: output singolo (es. 1 classificatore)
if isinstance(output, torch.Tensor):
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    print(f"Predizione: {pred_class} (prob: {probs[0][pred_class]:.2f})")

# Caso 2: output doppio (es. seme + numero)
elif isinstance(output, (list, tuple)) and len(output) == 2:
    output_seme, output_numero = output
    prob_n = F.softmax(output_numero, dim=1)
    prob_s = F.softmax(output_seme, dim=1)

    pred_n = torch.argmax(prob_n, dim=1).item()
    pred_s = torch.argmax(prob_s, dim=1).item()

    print(f"Predizione numero: {classi_numero[pred_n]} (prob: {prob_n[0][pred_n]:.2f})")
    print(f"Predizione seme: {classi_seme[pred_s]} (prob: {prob_s[0][pred_s]:.2f})")

else:
    raise ValueError("Formato output della rete non supportato")
