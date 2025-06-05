import os
from pathlib import Path
import sys

import cv2
import torch

from CNN_Double_head.DualHeadCNN import DualHeadCNN
from CNN_Double_head.predict_card_image import run_model as run_cnn_model
from application_fundamental_strategy import suggerisci_mossa

from ultralytics import YOLO
from torchvision import transforms

import argparse

PROJECT_DIR= os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carico i pesi del modello della CNN
#CNN_MODEL= DualHeadCNN()
#state_dict= torch.load(os.path.join(PROJECT_DIR, "CNN_Double_head/result/separated_layer_v1/weights.pth"), map_location= DEVICE)
#CNN_MODEL.load_state_dict(state_dict)

# Carico la CNN allenata
CNN_MODEL_PATH = "CNN_Double_head/result/separated_layer_v1/cnn_allenata_4.pth"
CNN_MODEL = torch.load(CNN_MODEL_PATH, map_location=DEVICE)

# Carico il modello allenato di YOLO
YOLO_MODEL= YOLO(os.path.join(PROJECT_DIR, "YOLO_cards_detector/runs/detect/yolo_finetuned_blackjack2/weights/best.pt"))
YOLO_MODEL.conf= 0.3    # Soglia minima di confidenza

# Trasformazioni immagine per la CNN
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# main com cam
def useCam():
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

# Main senza cam
def noCam(img_path: str):

    IMAGE_SPLITTER = 416//2 # Per differenziare le carte del dealer (in alto) da quelle in basso (del player)

    # === 2. Carica CNN
    CNN_MODEL.eval()

    # === 3. Carica immagine intera
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ottiengo la dimensione
    height, width, _ = img.shape

    # Disegno la linea centrale
    cv2.line(img, (0, (height//2)-1), (width-1, (height//2)-1), (255, 0, 255), 3)

    # Disegno le scritte "Dealer", "Player"
    cv2.putText(img, "Dealer", (5, height//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Player", (5, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # === 4. Applica YOLO
    results = YOLO_MODEL(img_rgb)
    print(results)
    
    # Inizializzo Array carte
    carta_dealer = ""
    carte_player = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped = img_rgb[y1:y2, x1:x2]
            # cv2.imshow("bounding box", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)

            # Preprocess
            input_tensor = transform(cropped).unsqueeze(0).to(DEVICE)

            # Predizione CNN
            with torch.no_grad():
                out_suit, out_num = CNN_MODEL(input_tensor)

            prob_n = torch.softmax(out_num, dim=1)
            prob_s = torch.softmax(out_suit, dim=1)
            pred_n = torch.argmax(prob_n, dim=1).item()
            pred_s = torch.argmax(prob_s, dim=1).item()

            # === Mappa predizione ad etichetta leggibile
            numero_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
            seme_labels = ['cuori', 'quadri', 'fiori', 'picche']
            etichetta = f"{numero_labels[pred_n]} di {seme_labels[pred_s]}"

            if (y1 < IMAGE_SPLITTER):
                carta_dealer = numero_labels[pred_n]

                # Disegna box e label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif(y1 < 300):
                carte_player.append(numero_labels[pred_n])

                # Disegna box e label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            print(f"[✓] Carta rilevata: {etichetta} @ [{x1}, {y1}, {x2}, {y2}]")


    print("carta Dealer: " + str(carta_dealer) + "\ncarte Player: " + str(carte_player))

    # === 5. Chiamo la funzione che applica la strategia fondamentale
    print(suggerisci_mossa(carte_player, carta_dealer))

    # === 6. Mostra immagine con carte etichettate
    cv2.imshow("Carte Rilevate", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


parser= argparse.ArgumentParser(description="YOLO card detector, this script start the detection on a passed image, the image can be one, a folder, or directly from the cam")
parser.add_argument("-o", "--output_path", type=str, help="output path where will be saved the results")

# Creo un gruppo di argomenti che si autoescludono
group= parser.add_mutually_exclusive_group(required= True)
group.add_argument("-s", "--screenshot_path", type=str, help="path of the image to be detected")
group.add_argument("-f", "--folder_path", type=str, help="path of the folder that contain the images to be detected")
group.add_argument("-c", "--use_cam", help="start the detection by using the webcam", action="store_true")


if __name__ == "__main__":
    # TODO Avviare cam con opencv2, fare screen del frame, dividere in due immagini
    #  img_sopra(dealer) img_sotto(player), passare le immagini a yolo, e poi le rilevazioni
    #  di yolo alla cnn, infine calcolare la mossa da fare e restituire il risultato a schermo

    args = parser.parse_args()

    o_path= ""
    if args.output_path is not None:
        o_path = f"{Path(args.output_path)}"

    if args.screenshot_path is not None:
        input_path= f"{Path(args.screenshot_path).resolve()}"
        noCam(input_path)

    elif args.use_cam:
        useCam()