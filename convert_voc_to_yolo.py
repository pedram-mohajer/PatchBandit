import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

def convert_annotation(voc_dir, image_id, output_label_dir):
    ann_path = os.path.join(voc_dir, "Annotations", f"{image_id}.xml")
    img_path = os.path.join(voc_dir, "JPEGImages", f"{image_id}.jpg")
    label_path = os.path.join(output_label_dir, f"{image_id}.txt")

    tree = ET.parse(ann_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(label_path, "w") as f:
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASS_TO_IDX:
                continue
            cls_id = CLASS_TO_IDX[cls]
            xml_box = obj.find("bndbox")
            xmin = float(xml_box.find("xmin").text)
            xmax = float(xml_box.find("xmax").text)
            ymin = float(xml_box.find("ymin").text)
            ymax = float(xml_box.find("ymax").text)
            # Normalize
            x = (xmin + xmax) / 2.0 / w
            y = (ymin + ymax) / 2.0 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            f.write(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    return img_path

def prepare_split(voc_dir, split_name, output_img_dir, output_label_dir, split_file):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    with open(split_file) as f:
        image_ids = [line.strip() for line in f.readlines()]

    for image_id in image_ids:
        src_img_path = convert_annotation(voc_dir, image_id, output_label_dir)
        dst_img_path = os.path.join(output_img_dir, f"{image_id}.jpg")
        shutil.copyfile(src_img_path, dst_img_path)

if __name__ == "__main__":
    voc_root = "VOCdevkit/VOC2007"
    output_root = "VOCYOLO"

    prepare_split(
        voc_dir=voc_root,
        split_name="train",
        output_img_dir=os.path.join(output_root, "images", "train"),
        output_label_dir=os.path.join(output_root, "labels", "train"),
        split_file=os.path.join(voc_root, "ImageSets", "Main", "trainval.txt")
    )

    prepare_split(
        voc_dir=voc_root,
        split_name="val",
        output_img_dir=os.path.join(output_root, "images", "val"),
        output_label_dir=os.path.join(output_root, "labels", "val"),
        split_file=os.path.join(voc_root, "ImageSets", "Main", "val.txt")
    )

    print("✅ VOC2007 successfully converted to YOLO format.")
