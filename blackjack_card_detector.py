import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO
from CNN_Double_head.cnn_card_classifier import DualHeadCNN  # definita nel tuo file
import matplotlib.pyplot as plt

# === CONFIG ===
YOLO_MODEL_PATH = "YOLO_cards_detector/runs/detect/yolo_finetuned_blackjack2/weights/best.pt"
CNN_MODEL_PATH = "CNN_Double_head/separate_layer_v2/cnn_allenata_5.pth"
IMAGE_PATH = "test_images_complete_net/test5.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Carica YOLO
yolo_model = YOLO(YOLO_MODEL_PATH)

# === 2. Carica CNN
cnn_model = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
cnn_model.eval()

# === 3. Trasformazioni immagine per la CNN
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# === 4. Carica immagine intera
img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === 5. Applica YOLO
results = yolo_model(img_rgb)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cropped = img_rgb[y1:y2, x1:x2]
        cv2.imshow("bounding box", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

        # Preprocess
        input_tensor = transform(cropped).unsqueeze(0).to(DEVICE)

        # Predizione CNN
        with torch.no_grad():
            out_num, out_suit = cnn_model(input_tensor)

        prob_n = torch.softmax(out_num, dim=1)
        prob_s = torch.softmax(out_suit, dim=1)
        pred_n = torch.argmax(prob_n, dim=1).item()
        pred_s = torch.argmax(prob_s, dim=1).item()

        # === Mappa predizione ad etichetta leggibile
        numero_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        seme_labels = ['cuori', 'quadri', 'fiori', 'picche']
        etichetta = f"{numero_labels[pred_s]} di {seme_labels[pred_n]}"

        print(f"[✓] Carta rilevata: {etichetta} @ [{x1}, {y1}, {x2}, {y2}]")

        # Disegna box e label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# === 6. Mostra immagine con carte etichettate
cv2.imshow("Carte Rilevate", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
