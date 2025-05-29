import os
import random

def reduce_dataset_in_place(base_dir, subset_name, keep_n, seed=42):
    random.seed(seed)
    image_dir = os.path.join(base_dir, subset_name, "images")
    label_dir = os.path.join(base_dir, subset_name, "labels")

    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(all_images) <= keep_n:
        print(f"{subset_name}: nulla da ridurre (n immagini = {len(all_images)})")
        return

    keep_images = set(random.sample(all_images, keep_n))

    for img_file in all_images:
        if img_file not in keep_images:
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)

    print(f"{subset_name}: ridotto a {keep_n} immagini")

# === ESEMPIO USO ===
base_path = "yolov5/data/blackjack"

reduce_dataset_in_place(base_path, "train", keep_n=4000)
reduce_dataset_in_place(base_path, "valid", keep_n=1000)
reduce_dataset_in_place(base_path, "test", keep_n=1000)
