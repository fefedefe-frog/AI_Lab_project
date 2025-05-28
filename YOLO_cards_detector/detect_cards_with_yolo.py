from ultralytics import YOLO
import cv2
import os

# === CONFIGURAZIONE ===
MODEL_PATH = "runs/detect/yolo_finetuned_blackjack2/weights/best.pt"  # <-- aggiorna se diverso
IMAGE_FOLDER = "test_images"  # Cartella con immagini da testare
OUTPUT_FOLDER = "test_results"     # Dove salvare i risultati
CONFIDENCE_THRESHOLD = 0.3    # Soglia minima di confidenza
RESIZE_DIM = (416, 416)  # Dimensione a cui ridimensionare le immagini

# === CREA CARTELLA RISULTATI SE NON ESISTE ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === CARICA MODELLO YOLO ===
model = YOLO(MODEL_PATH)
 
# === CICLO SULLE IMMAGINI DELLA CARTELLA ===
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        image_path = os.path.join(IMAGE_FOLDER, filename)

        # Carica e ridimensiona immagine
        img = cv2.imread(image_path)
        if img is None:
            print(f"[!] Errore nel leggere {filename}, salto.")
            continue

        img_resized = cv2.resize(img, RESIZE_DIM)

        # Inferenza
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, save=False)

        # Disegna i bounding box rilevati
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label = model.names[int(box.cls[0])]

                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (b[0], b[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Salva immagine con i box disegnati
        output_path = os.path.join(OUTPUT_FOLDER, f"pred_{filename}")
        cv2.imwrite(output_path, img)
        print(f"[✓] Salvato: {output_path}")
