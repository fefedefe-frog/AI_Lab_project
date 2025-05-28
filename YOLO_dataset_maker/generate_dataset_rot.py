import os
import math

import cv2
import random
import numpy as np


# Funzione usata per calcolare le dimensioni aggiornate di un rettangolo, che viene ruotato
def rotated_rect_size(width: int, height: int, angle_deg: int) -> tuple:
  rad= math.radians(angle_deg)
  cos_a= abs(math.cos(rad))
  sin_a= abs(math.sin(rad))
  new_width= int( width * cos_a + height * sin_a)
  new_height= int( width * sin_a + height * cos_a)
  return new_width, new_height


# Percorsi
INPUT_DIR_PATH = "input_images"
OUTPUT_DIR_PATH = "output_dataset/rotation"

# Dimensioni fisse
CARD_SIZE= (500, 726)
BG_SIZE= (1200, 1200)
BOX_SIZE= (120, 220)  # width, height


# numero di immagini per carta da generare
img_num_for_card= 50


# Lista file
cards = [card for card in os.listdir(os.path.join(INPUT_DIR_PATH, "cards"))]
backgrounds= [background for background in os.listdir(os.path.join(INPUT_DIR_PATH, "backgrounds"))]

# Cartella dove vengono salvate le immagini
images_output_dir_path= os.path.join(OUTPUT_DIR_PATH, "images")
os.makedirs(images_output_dir_path, exist_ok=True)

# Cartella dove vengono salvati i label delle immagini
labels_output_dir_path= os.path.join(OUTPUT_DIR_PATH, "labels")
os.makedirs(labels_output_dir_path, exist_ok=True)
for card_num, card in enumerate(cards):
  for count in range(1, img_num_for_card+1):
    print(f"\rCard: {card.split(".")[0]} di {card.split(".")[1]}\t|\tcard num:{card_num+1} out of {len(cards)}\t|\t\timage count: {count}/{img_num_for_card}", end="", flush=True)

    # Carico l'immagine della carta
    card_path = os.path.join(INPUT_DIR_PATH, "cards", card)
    card_rgba = cv2.imread(card_path, cv2.IMREAD_UNCHANGED) # Carico anche alpha channel per la trasparenza
    if card_rgba is None:
      print(f"⚠️ Immagine non caricata correttamente: {card_path}, salto...")
      continue

    # Viene ridimensionata la carta a 500x726 (anche se in teoria sono già tutte di questa dimensione)
    card_rgba = cv2.resize(card_rgba, (500, 726))

    # Divido i canali in rgb e alpha
    card_rgb= card_rgba[:, :, :3]
    card_alpha= card_rgba[:, :, 3] / 255.0  # alpha [0, 1]
    card_alpha= np.stack([card_alpha]* 3, axis=-1) # (H, W, 3)

    # Angolo di rotazione della carta random
    angle= random.randint(-30, 30)

    # Creo una nuova immagine che occupi lo spazio necessario a contenere l'immagine ruotata
    rot_card_size= rotated_rect_size(CARD_SIZE[0], CARD_SIZE[1], angle)
    rot_card_offset= ((rot_card_size[0] - CARD_SIZE[0]) //2, (rot_card_size[1] - CARD_SIZE[1]) //2)

    # Calcolo la matrice di rotazione dell'immagine
    rot_matrix= cv2.getRotationMatrix2D((CARD_SIZE[0] / 2, CARD_SIZE[1] / 2), angle, 1.0)
    rot_matrix[0, 2] += (rot_card_size[0] - CARD_SIZE[0]) / 2
    rot_matrix[1, 2] += (rot_card_size[1] - CARD_SIZE[1]) / 2

    # Inserisco la carta ruotata al centro dell'immagine
    rot_card_rgb= cv2.warpAffine(card_rgb, rot_matrix, rot_card_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rot_card_alpha= cv2.warpAffine(card_alpha, rot_matrix, rot_card_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    # Carico una immagine di background random tra quelle disponibili
    rand_bg= random.choice(backgrounds)
    bg_image= cv2.imread(os.path.join(INPUT_DIR_PATH, "backgrounds", rand_bg), cv2.IMREAD_UNCHANGED)
    bg_image= cv2.resize(bg_image, BG_SIZE)

    # Genero un offset (posizione) randomica da usare per piazzare casualmente la carta sullo sfondo
    offset = (
      random.randint(20, BG_SIZE[0] - rot_card_size[0] - 20),
      random.randint(20, BG_SIZE[1] - rot_card_size[1] - 20)
    )

    # Estraggo la regione di interesse dal background nella sezione dove andrà a inserirsi la carta
    roi = bg_image[offset[1]: offset[1] + rot_card_size[1], offset[0]: offset[0] + rot_card_size[0]]

    # Alpha blendig delle due imamgini
    blended= (rot_card_rgb * rot_card_alpha + roi * (1-rot_card_alpha)).astype(np.uint8)

    # Fondo le due immagini con applicati correttamente gli alpha
    bg_image[offset[1] : offset[1] + rot_card_size[1], offset[0] : offset[0] + rot_card_size [0]] = blended
    tras_point= (
      rot_matrix[0, 0] * 50 + rot_matrix[0, 1] * 100 + rot_matrix[0, 2],
      rot_matrix[1, 0] * 50 + rot_matrix[1, 1] * 100 + rot_matrix[1, 2]
    )

    # DEBUG: disegna il centro della bounding box di yolo
    # cv2.circle(bg_image, (int(tras_point[0]+ offset[0]), int(tras_point[1] + offset[1])), 5, (255, 0, 255), 1)

    # Vengono applicate modifiche di luminosità casuali
    brightness = random.uniform(0.8, 1.2)
    final = np.clip(bg_image * brightness, 0, 255).astype(np.uint8)

    # Viene salvata l'immagine
    card_output_full_path = os.path.join(images_output_dir_path, f"{count}_{card}")
    cv2.imwrite(card_output_full_path, final)

    # Calcolo le informazioni della bounding box di yolo:
    # - Applicare traslazione del punto centrale della bounding box
    # - Applicare offset del punto traslato
    # - Ricalcolo delle dimensioni della bounding box dovuto alla rotazione della carta

    rot_box_size= rotated_rect_size(BOX_SIZE[0], BOX_SIZE[1], angle)
    resize_percentage= 1
    if abs(angle) > 8: # Se l'immagine è molto inclinata si avrebbe una bounding box abbastanza larga rispetto a quello che dovrebbe contenere
      resize_percentage= 0.8

    bounding_box_info = (
      0,  # class_id
      (offset[0] + tras_point[0]) / BG_SIZE[0],
      # center_x: parto usando la coordinata X dell'offset della carta, gli sommo la metà della larghezza della box, e per la normalizzazione [0, 1] divido tutto per la larghezza dell'immagine completa
      (offset[1] + tras_point[1]) / BG_SIZE[1],
      # center_y: stessa cosa della coordinata x ma usando la Y dell'offset e la lunghezza della box e dell'immagine completa
      rot_box_size[0]*resize_percentage / BG_SIZE[0],
      rot_box_size[1]*resize_percentage / BG_SIZE[1]
    )

    with open(os.path.join(labels_output_dir_path, f"{count}_{card.replace(".png", ".txt")}"), "w") as f:
       f.write(" ".join([str(round(x, 6)) for x in bounding_box_info]))
print("\n✅ Dataset creato con OpenCV!")