from ultralytics import YOLO

def main():
    model = YOLO("../../YOLO_cards_detector/yolov8s.pt")
    model.train(
        data="./yolov5/data/blackjackDataset/dataset1/data.yaml",
        epochs=30,
        imgsz=416,
        batch=16,
        name="yolo_s_30epochs"
    )

if __name__ == "__main__":
    main()

