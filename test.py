from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from dataset import VOCDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomAffine, ColorJitter, ToTensor, Normalize
import torch.optim as optim
import argparse
from tqdm.autonotebook import tqdm
# from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import cv2


def get_args():
    parser = argparse.ArgumentParser(description="Detection")
    parser.add_argument("--image_path", default="test_images/test.jpg", type=str, help="path to test image")
    parser.add_argument("--image_size", default=416, type=int)
    parser.add_argument("--conf_threshold", default=0.5, type=float)
    parser.add_argument("--checkpoint", default="trained_models/last.pt", type=str, help="path to model checkpoint file")

    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    categories = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                  "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                  "sheep", "sofa", "train", "tvmonitor"]

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=len(categories))
    model.to(device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))
    images = [torch.from_numpy(image).to(device).float()]

    print(images[0].shape)

    with (torch.no_grad()):
        predictions = model(images)
        print(predictions)

    for box, score, label in zip(predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]):
        if (score > args.conf_threshold):
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin / args.image_size * width)
            ymin = int(ymin / args.image_size * height)
            xmax = int(xmax / args.image_size * width)
            ymax = int(ymax / args.image_size * height)

            cv2.rectangle(ori_image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
            cv2.putText(ori_image, text=categories[label] + "{:.2f}".format(score), org=(xmin, ymin - 10), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(255, 255, 0), thickness=1)

    cv2.imwrite("prediction.jpg", ori_image)




if __name__ == "__main__":
    args = get_args()
    test(args)