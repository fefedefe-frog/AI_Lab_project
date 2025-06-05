import os
import cv2

# === CONFIGURAZIONE ===
INPUT_FOLDER = "./simple_card_images"
DIMENSIONE_CROP = (160, 65)  # (altezza, larghezza)
OFFSET_X, OFFSET_Y = 3, 5   # margine dal bordo sinistro e alto

# === Mostra tutte le immagini con bounding box
for nome_file in sorted(os.listdir(INPUT_FOLDER)):
    if not nome_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(INPUT_FOLDER, nome_file)
    img = cv2.imread(path)

    if img is None:
        print(f"Errore su {nome_file}")
        continue

    # Disegna rettangolo del crop
    x1, y1 = OFFSET_X, OFFSET_Y
    x2, y2 = x1 + DIMENSIONE_CROP[1], y1 + DIMENSIONE_CROP[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Controllo bounding box", img)
    print(f"Mostrando: {nome_file}")
    key = cv2.waitKey(0)

    if key == ord('q'):
        break  # premi 'q' per uscire

cv2.destroyAll
