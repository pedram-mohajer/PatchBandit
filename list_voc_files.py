# list_voc_files.py

import os

def list_first_n(root, n=40):
    """
    Walk through each immediate subfolder of root,
    list and sort its files, then print the first n.
    """
    for sub in sorted(os.listdir(root)):
        subdir = os.path.join(root, sub)
        if not os.path.isdir(subdir):
            continue
        print(f"\n=== {subdir} ===")
        files = sorted(os.listdir(subdir))
        for fname in files[:n]:
            print(f"  {fname}")
        if len(files) > n:
            print(f"  ... ({len(files)} total)")

if __name__ == "__main__":
    voc_root = os.path.join('.', 'VOCdevkit', 'VOC2007')
    if not os.path.isdir(voc_root):
        print(f"ERROR: {voc_root} not found")
        exit(1)
    # list subfolders JPEGImages, Annotations, ImageSets/Main, SegmentationClass, etc.
    list_first_n(voc_root)
    # also drill into ImageSets/Main
    main_splits = os.path.join(voc_root, 'ImageSets', 'Main')
    if os.path.isdir(main_splits):
        print(f"\n=== {main_splits} ===")
        for fname in sorted(os.listdir(main_splits))[:40]:
            print(f"  {fname}")
        print(f"  ... ({len(os.listdir(main_splits))} total)")
