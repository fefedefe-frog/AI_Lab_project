import argparse
import yolo_dataset_maker.generate_dataset as generator

parser= argparse.ArgumentParser(description="YOLO dataset maker, this script automatically generate the images and labels for yolo")
parser.add_argument("image_num", type=int, help="how many images to generate")
parser.add_argument("-o", "--output_path", type=str, help="output directory path where will be saved the dataset")
parser.add_argument("-r", "--rotation-on", help="enable random rotation of the generated images", action="store_true")


if __name__ == "__main__":
    args= parser.parse_args()

    generator.generate(args.rotation_on, args.image_num, args.output_path)