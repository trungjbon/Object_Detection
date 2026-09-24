from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from dataset import VOCDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ColorJitter, ToTensor, Normalize
import torch.optim as optim
import argparse
from tqdm.autonotebook import tqdm
import os
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

def collate_fn(batch):
    all_images = []
    all_labels = []

    for image, label in batch:
        all_images.append(image)
        all_labels.append(label)

    return all_images, all_labels


def get_args():
    parser = argparse.ArgumentParser(description="Detection")
    parser.add_argument("--data_path", default="data/voc", type=str, help="the root folder of the data")
    parser.add_argument("--epochs", default=50, type=int, help="Total number of epochs")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--image_size", default=416, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--es_min_delta", default=0.0, type=float,
                        help="Early stopping's parameters: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", default=0, type=int,
                        help="Early stopping's parameters: number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--checkpoint", default=None, type=str, help="path to model checkpoint file")
    parser.add_argument("--log_path", default="tensorboard/pascal_voc", type=str)
    parser.add_argument("--save_path", default="trained_models", type=str)

    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    train_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    val_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_set = VOCDataset(root=args.data_path, year="2007", image_set="train", download=False, transform=train_transform)
    val_set = VOCDataset(root=args.data_path, year="2007", image_set="val", download=False, transform=val_transform)

    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 6,
        "collate_fn": collate_fn
    }
    val_params = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 6,
            "collate_fn": collate_fn
        }
    
    train_dataloader = DataLoader(train_set, **train_params)
    val_dataloader = DataLoader(val_set, **val_params)

         
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    print(model.roi_heads.box_predictor)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=len(train_set.categories))
    print(model.roi_heads.box_predictor)

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w)

    if (not os.path.isdir(args.save_path)):
        os.makedirs(args.save_path)

    best_map = -1

    for epoch in range(args.epochs):
        # Training Step
        model.train()
        train_loss = []
        train_progress_bar = tqdm(train_dataloader, colour="cyan")

        for iter, (images, labels) in enumerate(train_progress_bar):
            images = [image.to(device) for image in images]
            labels = [{"boxes": label["boxes"].to(device), "labels": label["labels"].to(device)} for label in labels]

            loss_components = model(images, labels)
            losses = sum([loss for loss in loss_components.values()])

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss.append(losses.item())
            avg_loss = np.mean(train_loss)

            train_progress_bar.set_description("Epoch {}/{}. Loss {:.4f}".format(epoch + 1, args.epochs, avg_loss))

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        # # Evaluation Step
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        val_progress_bar = tqdm(val_dataloader, colour="cyan")
        
        for iter, (images, labels) in enumerate(val_progress_bar):
            images = [image.to(device) for image in images]
            labels = [{"boxes": label["boxes"].to(device), "labels": label["labels"].to(device)} for label in labels]

            with (torch.no_grad()):
                predictions = model(images)
                metric.update(predictions, labels)

        map = metric.compute()

        if (map["map"] > best_map):
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            best_map = map["map"]

        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))


if __name__ == "__main__":
    args = get_args()
    train(args)