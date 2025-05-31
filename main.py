import os
import sys

import cv2
import torch

from CNN_Double_head.DualHeadCNN import DualHeadCNN
from CNN_Double_head.predict_card_image import run_model as run_cnn_model

from ultralytics import YOLO

PROJECT_DIR= os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carico i pesi del modello della CNN
CNN_MODEL= DualHeadCNN()
state_dict= torch.load(os.path.join(PROJECT_DIR, "CNN_Double_head/result/separated_layer_v1/weights.pth"), map_location= DEVICE)
CNN_MODEL.load_state_dict(state_dict)

# Carico il modello allenato di YOLO
YOLO_MODEL= YOLO(os.path.join(PROJECT_DIR, "YOLO_cards_detector/runs/detect/yolo_finetuned_blackjack2/weights/best.pt"))
YOLO_MODEL.conf= 0.3    # Soglia minima di confidenza

if __name__ == "__main__":
    # TODO Avviare cam con opencv2, fare screen del frame, dividere in due immagini
    #  img_sopra(dealer) img_sotto(player), passare le immagini a yolo, e poi le rilevazioni
    #  di yolo alla cnn, infine calcolare la mossa da fare e restituire il risultato a schermo

    # Avvio la webcam (0 è la cam di default
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Errore nell'apertura della videocamera")

    while True:
        # Catturo il frame e una variabile che indica se il frame è
        # catturato correttamente
        ret, frame = cam.read()

        if not ret:
            break

        # Passo il frame ricevuto al modello, per la predizione
        results = YOLO_MODEL(frame, save=False)[0]

        # Per ogni risultato di yolo eseguo la cnn
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cropped = frame[y1:y2, x1:x2]
                cv2.imshow("bounding box", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)

                # Predizione CNN
                pred_seme, pred_numero= run_cnn_model(cropped, CNN_MODEL)

                # Preparo l'etichetta da mostrare a schermo
                etichetta = f"{pred_numero[0]}({pred_numero[1]*100:.2f}%) di {pred_seme[0]}({pred_seme[1]*100:.2f}%)"

                print(f"[✓] Carta rilevata: {etichetta} @ [{x1}, {y1}, {x2}, {y2}]")

                # Disegna box e label sull'immagine ricevuta in input
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Model detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break