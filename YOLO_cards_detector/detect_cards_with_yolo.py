import argparse
from pathlib import Path
from typing import Any

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
def run_model_on_folder(folder: str, output_folder: str) -> None:
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_path = os.path.join(folder, filename)


            '''
            # Carica e ridimensiona immagine
            img = cv2.imread(image_path)
            if img is None:
                print(f"[!] Errore nel leggere {filename}, salto.")
                continue
            img_resized = cv2.resize(img, RESIZE_DIM)
            # Inferenza
            results = model(img_resized, conf=CONFIDENCE_THRESHOLD, save=False)
            

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
            '''

            ann_img= run_model(image_path)
            if ann_img is None:
                print("unable to predict")
                continue


            # Salva immagine con i box disegnati
            output_path = os.path.join(output_folder, f"pred_{filename}")
            cv2.imwrite(output_path, ann_img)
            print(f"[✓] Salvato: {output_path}")


def run_model(image_path: str, output_path: str= None) -> Any | None:

    # Carica e ridimensiona immagine
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Errore nel leggere {image_path}, salto.")
        return None
    img_resized = cv2.resize(image, RESIZE_DIM)


    results = model(img_resized, conf=CONFIDENCE_THRESHOLD, save=False)[0]

    annotated_image= results.plot()

    if output_path is not None:
        cv2.imwrite(os.path.join(output_path, f"result.{image_path.split('0')[-1]}"), annotated_image)
    return annotated_image


def run_model_on_cam() -> None:
    # Avvio la webcam (0 è la cam di default
    cam= cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Errore nell'apertura della videocamera")

    while True:
        # Catturo il frame e una variabile che indica se il frame è
        # catturato correttamente
        ret, frame = cam.read()

        if not ret:
            break

        # Passo il frame ricevuto al modello, per la predizione
        results = model(frame, conf=CONFIDENCE_THRESHOLD, save=False)[0]

        # Disegno la bounding box della carta individuata sul frame
        annotated_frame= results.plot()

        cv2.imshow("Model detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


parser= argparse.ArgumentParser(description="YOLO card detector, this script start the detection on a passed image, the image can be one, a folder, or directly from the cam")
parser.add_argument("-o", "--output_path", type=str, help="output path where will be saved the results")

# Creo un gruppo di argomenti che si autoescludono
group= parser.add_mutually_exclusive_group(required= True)
group.add_argument("-i", "--image_path", type=str, help="path of the image to be detected")
group.add_argument("-f", "--folder_path", type=str, help="path of the folder that contain the images to be detected")
group.add_argument("-c", "--use_cam", help="start the detection by using the webcam", action="store_true")



if __name__ == "__main__":
    args = parser.parse_args()

    o_path= OUTPUT_FOLDER
    if args.output_path is not None:
        o_path = f"{Path(args.output_path)}"

    if args.image_path is not None:
        i_path= f"{Path(args.image_path).resolve()}"
        run_model(i_path, o_path)

    elif args.folder_path is not None:
        folder_path= f"{Path(args.folder_path).resolve()}"
        run_model_on_folder(folder_path, o_path)

    elif args.use_cam:
        run_model_on_cam()