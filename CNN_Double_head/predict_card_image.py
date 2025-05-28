import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# === CONFIGURAZIONE ===
MODEL_PATH = "modello_addestrato.pth"      # Path al tuo modello
IMAGE_PATH = "immagine_di_test.jpg"        # Immagine da classificare
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Trasformazioni immagine (modifica se servono altre) ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),         # stessa dimensione usata in training
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])     # o [0.5, 0.5, 0.5] per RGB
])

# === Carica immagine ===
if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"{IMAGE_PATH} non trovato")

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)  # shape: (1, C, H, W)

# === Carica modello ===
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

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
    output_numero, output_seme = output
    prob_n = F.softmax(output_numero, dim=1)
    prob_s = F.softmax(output_seme, dim=1)

    pred_n = torch.argmax(prob_n, dim=1).item()
    pred_s = torch.argmax(prob_s, dim=1).item()

    print(f"Predizione numero: {pred_n} (prob: {prob_n[0][pred_n]:.2f})")
    print(f"Predizione seme: {pred_s} (prob: {prob_s[0][pred_s]:.2f})")

else:
    raise ValueError("Formato output della rete non supportato")
