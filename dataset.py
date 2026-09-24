from torchvision.datasets import VOCDetection
from pprint import pprint
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ColorJitter, ToTensor, Normalize
import torch


class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                           "sheep", "sofa", "train", "tvmonitor"]

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        image, targets = super().__getitem__(index)
        old_h = int(targets["annotation"]["size"]["height"])
        old_w = int(targets["annotation"]["size"]["width"])
        _, new_h, new_w = image.shape
        targets = targets["annotation"]["object"]

        labels = []
        bboxes = []
        output = {}

        for target in targets:
            label = target["name"]
            labels.append(self.categories.index(label))

            xmin = int(float(target["bndbox"]["xmin"]) / old_w * new_w)
            ymin = int(float(target["bndbox"]["ymin"]) / old_h * new_h)
            xmax = int(float(target["bndbox"]["xmax"]) / old_w * new_w)
            ymax = int(float(target["bndbox"]["ymax"]) / old_h * new_h)
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)

        output["boxes"] = torch.FloatTensor(bboxes)
        output["labels"] = torch.LongTensor(labels)

        return image, output



if __name__ == "__main__":
    train_transform = Compose([
            Resize((416, 416)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
    dataset = VOCDataset(root="data/voc", year="2007", image_set="trainval", download=False, transform=train_transform)
    index = 225

    image, target = dataset.__getitem__(index)

    print(image.shape)
    print(target)
    
    # image = np.array(image)
    # print(image.shape)
    # image = np.transpose(image, (1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # print(bboxes)
    # print(labels)

    # for bbox in bboxes:
    #     xtl, ytl, xbr, ybr = bbox
    #     cv2.rectangle(image, (xtl, ytl), (xbr, ybr), (0, 255, 0), 1)
    # cv2.imwrite("sample.jpg", image)