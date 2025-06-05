import argparse
from pathlib import Path
from typing import Union, Tuple

import torch
import torch.nn.functional as functional
from torchvision import transforms
import os
import cv2

from CNN_Double_head import DualHeadCNN

# === CONFIGURAZIONE ===
MODEL_PATH = "cnn_allenata.pth"      # Path al tuo modello
IMAGE_PATH = "test_images/test3.png"        # Immagine da classificare
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisco le classi per seme e numeri definiti come in CardDataset
CLASSI_SEME: tuple= ("cuori", "quadri", "fiori", "picche")
CLASSI_NUMERO: tuple= ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "K", "Q", "J")


# === Trasformazioni immagine (modifica se servono altre) ===
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 180)),
    transforms.ToTensor(),  # converte e normalizza da [0–255] → [0–1]
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # ora [0–1] → [-1, 1]
])


def run_model(image, model_to_use) -> Union[Tuple[Tuple[str, float], Tuple[str, float]], Tuple[str, float]]:

    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)  # shape: (1, C, H, W)

    # === Predizione ===
    with torch.no_grad():
        output = model_to_use(input_tensor)

    # === Interpreta output
    # Caso 1: output singolo (es. 1 classificatore)
    if isinstance(output, torch.Tensor):
        probs = functional.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        return f"{pred_class}", float(probs[0][pred_class])

    # Caso 2: output doppio (es. seme + numero)
    elif isinstance(output, (list, tuple)) and len(output) == 2:
        output_seme, output_numero = output
        prob_n = functional.softmax(output_numero, dim=1)
        prob_s = functional.softmax(output_seme, dim=1)

        pred_n = torch.argmax(prob_n, dim=1).item()
        pred_s = torch.argmax(prob_s, dim=1).item()

        return (CLASSI_SEME[pred_s], float(prob_s[0][pred_s])), (CLASSI_NUMERO[pred_n], float(prob_n[0][pred_n]))

    else:
        raise ValueError("Formato output della rete non supportato")

parser= argparse.ArgumentParser(description="CNN card value detector, this script start the detection on a passed image")
parser.add_argument("input_image", type=str, help="input path of the image to predict")
parser.add_argument("-m", "--model_path", type=str, help="path of the model to load and use for the prediction")
parser.add_argument("-w", "--weights_only", help="define if the model to load are only it's weights", action="store_true")

if __name__ == "__main__":
    args= parser.parse_args()

    input_path= IMAGE_PATH
    if args.input_image is not None:
        input_path = f"{Path(args.input_image).resolve()}"

    model_path = MODEL_PATH
    if args.model_path is not None:
        model_path= f"{Path(args.model_path).resolve()}"


    # === Carica immagine ===
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"{input_path} non trovato")

    

    # === Carica modello ===
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.eval()

    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result= run_model(img, model)
    if isinstance(result[0], tuple):
        print(f"Predizione numero: {result[0][0]} (prob: {result[0][1]:.2f})")
        print(f"Predizione seme: {result[1][0]} (prob: {result[1][1]:.2f})")
    else:
        print(f"Predizione: {result[0]} (prob: {result[1]:.2f})")
