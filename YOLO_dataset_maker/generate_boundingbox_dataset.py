import os
import cv2
from ultralytics import YOLO
import torch

# === CONFIGURAZIONE ===
CARTELLA_IMMAGINI = "../YOLO_cards_detector/mani_blackjack"
MODELLO_YOLO_PATH = "../YOLO_cards_detector/runs/detect/yolo_finetuned_blackjack2/weights/best.pt"
CARTELLA_OUTPUT = "../cnn_dataset_maker/boundingBox_dataset"
CONFIDENCE_THRESHOLD = 0.3  # opzionale: scarta box con confidenza troppo bassa

# === CREAZIONE CARTELLA OUTPUT ===
os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

# === CARICA MODELLO YOLO ===
model = YOLO(MODELLO_YOLO_PATH)

# === INDICE PER NOMINARE I FILE RITAGLIATI ===
ritaglio_counter = 0

# === SCORRI TUTTE LE IMMAGINI DELLA CARTELLA ===
for nome_file in sorted(os.listdir(CARTELLA_IMMAGINI)):
    if not nome_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path_img = os.path.join(CARTELLA_IMMAGINI, nome_file)
    img = cv2.imread(path_img)

    if img is None:
        print(f"[!] Errore nel leggere l'immagine: {nome_file}")
        continue

    # Applica YOLO
    results = model(img)[0]

    for box in results.boxes:
        score = box.conf.item()
        if score < CONFIDENCE_THRESHOLD:
            continue

        # Ottieni coordinate bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Ritaglia e salva
        crop = img[y1:y2, x1:x2]
        nome_ritaglio = f"img_{ritaglio_counter}.jpg"
        path_ritaglio = os.path.join(CARTELLA_OUTPUT, nome_ritaglio)
        cv2.imwrite(path_ritaglio, crop)
        ritaglio_counter += 1

        print(f"[✔] Salvato: {nome_ritaglio}")

print(f"\n✅ Completato. Totale ritagli salvati: {ritaglio_counter}")
