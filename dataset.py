import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class AircraftDataset(Dataset):
    def __init__(self, dataset_dir, class_to_idx, transforms=None):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.class_to_idx = class_to_idx

        # Get list of image files
        self.image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Image file
        img_name = self.image_files[idx]
        img_path = os.path.join(self.dataset_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Corresponding annotation CSV file
        annot_name = img_name.replace('.jpg', '.csv')
        annot_path = os.path.join(self.dataset_dir, annot_name)

        # Check if annotation file exists
        if not os.path.exists(annot_path):
            # No annotations for this image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            annotations = pd.read_csv(annot_path)

            # Get bounding boxes and labels
            boxes = []
            labels = []
            for _, row in annotations.iterrows():
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']
                boxes.append([xmin, ymin, xmax, ymax])

                class_name = row['class']
                labels.append(self.class_to_idx[class_name])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if self.transforms:
            image = self.transforms(image)

        return image, target
