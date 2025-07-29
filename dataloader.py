# dataloader.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# PASCAL VOC 2007 class names
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

class PascalVOCLoader:
    """
    Data loader for PASCAL VOC 2007 detection.
    Yields (image_tensor [3,H,W], boxes [N,4], labels [N], annotation dict).

    Assumes folder structure under 'root':
      VOCdevkit/VOC2007/
        JPEGImages/  (images)
        Annotations/ (XML)
        ImageSets/Main/ (split lists)
    """

    def __init__(
        self,
        root: str = '.',            # project root containing VOCdevkit/
        year: str = "2007",
        image_set: str = "trainval",
        inp_size: int = 416,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        # Transform: resize, to-tensor, normalize
        self.transform = T.Compose([
            T.Resize((inp_size, inp_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # VOCDetection expects root/VOCdevkit/VOC{year}/...
        self.dataset = VOCDetection(
            root,
            year=year,
            image_set=image_set,
            download=False,
            transform=self.transform,
        )

        # DataLoader: batch_size=1 for individual images
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._iter = iter(self.loader)

    def next(self):
        """
        Get next sample: image, target boxes, labels, and raw annotation.
        Returns:
          img_tensor: torch.FloatTensor [3,H,W]
          boxes: torch.FloatTensor [N,4] in (xmin,ymin,xmax,ymax)
          labels: torch.LongTensor [N]
          anno: raw annotation dict
        """
        try:
            img_batch, target = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            img_batch, target = next(self._iter)

        # Unwrap batch dimension
        img = img_batch.squeeze(0)  # [3,H,W]
        anno = target['annotation']

        # Extract object list
        objs = anno.get('object', [])
        if isinstance(objs, dict):
            objs = [objs]

        boxes = []
        labels = []
        for obj in objs:
            bbox = obj['bndbox']
            # Handle potential lists in XML parsing
            def get_val(key):
                raw = bbox.get(key)
                if isinstance(raw, (list, tuple)):
                    return raw[0]
                return raw
            xmin = int(float(get_val('xmin')))
            ymin = int(float(get_val('ymin')))
            xmax = int(float(get_val('xmax')))
            ymax = int(float(get_val('ymax')))
            boxes.append([xmin, ymin, xmax, ymax])

            name_raw = obj.get('name')
            name = name_raw[0] if isinstance(name_raw, (list, tuple)) else name_raw
            name = name.lower().strip()
            labels.append(CLASS_TO_IDX.get(name, -1))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return img, boxes, labels, anno

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    loader = PascalVOCLoader(
        root='.',
        year="2007",
        image_set="trainval",
        inp_size=416,
        num_workers=2,
        shuffle=True,
    )

    img_tensor, boxes, labels, annotation = loader.next()

    print("Image tensor shape:", tuple(img_tensor.shape))
    print("Boxes:", boxes)
    print("Labels:", labels)
    print("Raw annotation keys:", list(annotation.keys()))

    # Visualize image and boxes
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_vis = (img_tensor * std + mean).permute(1,2,0).numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(np.clip(img_vis, 0, 1))
    for (xmin, ymin, xmax, ymax), lbl in zip(boxes, labels):
        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                          edgecolor='r', facecolor='none', linewidth=2)
        )
        if lbl >= 0:
            plt.text(xmin, ymin-5, VOC_CLASSES[lbl], color='yellow', fontsize=8)
    plt.title("Sample with Bounding Boxes")
    plt.axis('off')
    plt.show()


