from ultralytics import YOLO

def main():

    # Carico il modello pre-allenato
    model = YOLO("runs/detect/yolo_s_30epochs3/weights/best.pt") 

    # Avvia il fine-tuning
    model.train(
        data="blackjackDataset/dataset2/data.yaml",
        epochs=20,              
        imgsz=416,
        batch=16,                
        name="yolo_finetuned_blackjack"
    )

if __name__ == "__main__":
    main()


