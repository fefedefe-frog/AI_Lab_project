import os
import sys
from pathlib import Path
from typing import Any

import cv2
import gradio as gr


from CNN_Double_head.DualHeadCNN import DualHeadCNN
from CNN_Double_head.predict_card_image import run_model
from application_fundamental_strategy import suggerisci_mossa

import torch
from torchvision import transforms
from ultralytics import YOLO

import argparse

PROJECT_DIR= os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carico i pesi del modello della CNN
#CNN_MODEL= DualHeadCNN()
#state_dict= torch.load(os.path.join(PROJECT_DIR, "CNN_Double_head/result/separated_layer_v1/weights.pth"), map_location= DEVICE)
#CNN_MODEL.load_state_dict(state_dict)

# Carico la CNN allenata
CNN_MODEL_PATH = "CNN_Double_head/result/separated_layer_v2/cnn_allenata_5.pth"
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
def use_cam() -> None:
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
                pred_seme, pred_numero= run_model(cropped, CNN_MODEL)

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
def no_cam(img_path: str) -> None:

    # === 1. Carica immagine intera
    img = cv2.imread(img_path)
    img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # === 2. Carica CNN
    CNN_MODEL.eval()

    # Ottiengo la dimensione
    height, width, _ = img.shape

    # Disegno la linea centrale
    cv2.line(img, (0, (height // 2) - 1), (width - 1, (height // 2) - 1), (255, 0, 255), 3)

    # Disegno le scritte "Dealer", "Player"
    cv2.putText(img, "Dealer", (5, height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Player", (5, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # === 2. Applica YOLO
    results = YOLO_MODEL(img_rgb)

    # Inizializzo Array carte
    carta_dealer = ""
    carte_player = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped = img_rgb[y1:y2, x1:x2]

            # Uso la CNN per predirre il valore della carta
            pred_seme, pred_numero = run_model(cropped,
                                               CNN_MODEL)  # Due tuple, restituiscono la predizione con la percentuale di predizione

            etichetta = f"{pred_numero[0]} di {pred_seme[0]}"

            if y1 < height//2:
                carta_dealer = pred_numero[0]

                # Disegna box e label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif y1 < 300:
                carte_player.append(pred_numero[0])

                # Disegna box e label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(f"[✓] Carta rilevata: {etichetta} @ [{x1}, {y1}, {x2}, {y2}]")

    print("carta Dealer: " + str(carta_dealer) + "\ncarte Player: " + str(carte_player))

    # === 3. Chiamo la funzione che applica la strategia fondamentale
    mossa: str = suggerisci_mossa(carte_player, carta_dealer)
    print("mossa: " + str(mossa))
    cv2.putText(img, f"Sugg.: {mossa}", (width // 2, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(125, 125, 125), 2)
    cv2.imshow("Model detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_evalutation(img_rgb) -> Any:
    # === 1. Carica CNN
    CNN_MODEL.eval()

    # Ottiengo la dimensione
    height, width, _ = img_rgb.shape

    # === 2. Applica YOLO
    results = YOLO_MODEL(img_rgb)
    
    # Inizializzo Array carte
    carta_dealer = ""
    carte_player = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped = img_rgb[y1:y2, x1:x2]

            # Uso la CNN per predirre il valore della carta
            pred_seme, pred_numero= run_model(cropped, CNN_MODEL)   # Due tuple, restituiscono la predizione con la percentuale di predizione

            etichetta = f"{pred_numero[0]} di {pred_seme[0]}"

            if y1 < height//2:
                carta_dealer = pred_numero[0]

                # Disegna box e label
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 125, 0), 2)
                cv2.putText(img_rgb, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 0), 2)
            elif y1 < height - 300:
                carte_player.append(pred_numero[0])

                # Disegna box e label
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 125, 0), 2)
                cv2.putText(img_rgb, etichetta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 0), 2)
                    
            print(f"[✓] Carta rilevata: {etichetta} @ [{x1}, {y1}, {x2}, {y2}]")

    print("carta Dealer: " + str(carta_dealer) + "\ncarte Player: " + str(carte_player))

    # === 3. Chiamo la funzione che applica la strategia fondamentale
    mossa: str= suggerisci_mossa(carte_player, carta_dealer)
    print("mossa: " + str(mossa))

    dealer_img= img_rgb[:height//2, :, :]
    player_img= img_rgb[height//2+1:, :, :]
    return (
        dealer_img,
        player_img,
        carta_dealer,
        ', '.join(carte_player),
        mossa
    )


def use_gradio() -> None:

    # creo la gui tramite i blocchi di gradio
    with gr.Blocks() as demo:
        gr.Markdown("BlackJack game helper")

        # La prima riga conterrà le due sezioni per le immagini, una di input e una di output
        with gr.Row():
            with gr.Column():
                # Permette il caricamento di immagini sia tramite immagini dirette, che tramite frame della camera
                input_image = gr.Image(sources=["webcam", "upload"], type="numpy", label= "Input")

                btn = gr.Button("Run Predict")

            with gr.Column():
                with gr.Column():
                    dealer_card = gr.Textbox(label= "Carta Dealer", interactive= False)
                    dealer_output = gr.Image(label= "Dealer Image")

                with gr.Column():
                    player_output = gr.Image(label= "Player Image")
                    with gr.Row():
                        player_cards = gr.Textbox(label= "Carte Player", interactive= False)
                        mossa_text = gr.Textbox(label= "Mossa Consigliata", interactive= False)

        btn.click(
            fn=run_evalutation,
            inputs=input_image,
            outputs=[
                dealer_output,
                player_output,
                dealer_card,
                player_cards,
                mossa_text
            ])

    demo.launch()

parser= argparse.ArgumentParser(description="YOLO card detector, this script start the detection on a passed image, the image can be one, a folder, or directly from the cam")

# Creo un gruppo di argomenti che si autoescludono
group= parser.add_mutually_exclusive_group(required= True)
group.add_argument("-s", "--screenshot_path", type=str, help="path of the image to be detected")
group.add_argument("-g", "--use_gradio", help="path of the folder that contain the images to be detected", action="store_true")
group.add_argument("-c", "--use_cam", help="start the detection by using the webcam", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.screenshot_path is not None:
        input_path= f"{Path(args.screenshot_path).resolve()}"
        no_cam(input_path)

    elif args.use_gradio is not None:
        use_gradio()

    elif args.use_cam is not None:
        use_cam()