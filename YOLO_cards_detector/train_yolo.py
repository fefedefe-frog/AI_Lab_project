import argparse
from ultralytics import YOLO

def train(data_path: str="./yolov5/data/blackjackDataset/dataset1/data.yaml", epochs: int=10, imgsz: int= 416, batch_size: int= 16, output_folder_name: str= "./yolo_output"):
    model = YOLO("yolov8s.pt")
    model.train(
        data= data_path,
        epochs= epochs,
        imgsz= imgsz,
        batch= batch_size,
        name= output_folder_name
    )

parser= argparse.ArgumentParser(description="YOLO trainer script, this script launch the train of yolo")
parser.add_argument("data_path", type=str, help="path to the data.yaml file")
parser.add_argument("epochs", type=int, help="how many epochs the model will train")
parser.add_argument("-b", "--batch_size", type=int, help="the size of the batch")
parser.add_argument("-i", "--img_sz", type=int, help="how yolo will resize the image from the dataset")
parser.add_argument("-o", "--output_folder_name", type=str, help="name of the output folder")


if __name__ == "__main__":
    args= parser.parse_args()

    passed_args= {k: v for k, v in vars(args).items() if v is not None}
    train(**passed_args)
