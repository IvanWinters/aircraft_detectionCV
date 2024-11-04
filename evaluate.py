import torch
from torch.utils.data import DataLoader
from model import get_model
from dataset import AircraftDataset
from utils import get_transform, class_to_idx
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.models.detection import FasterRCNN
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model, data_loader, device):
    model.eval()
    metric = 0.0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", unit="batch"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            # Example: Calculate Intersection over Union (IoU) for simplicity
            for output, target in zip(outputs, targets):
                if len(output['boxes']) == 0 or len(target['boxes']) == 0:
                    continue
                ious = box_iou(output['boxes'], target['boxes'])
                max_ious, _ = ious.max(dim=1)
                metric += max_ious.mean().item()
                total += 1

    if total > 0:
        average_iou = metric / total
        print(f"Average IoU: {average_iou:.4f}")
    else:
        print("No overlapping boxes found.")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = len(class_to_idx)
    model = get_model(num_classes)
    model.to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load('aircraft_detector.pth', map_location=device))

    dataset_dir = 'data/dataset'
    dataset = AircraftDataset(
        dataset_dir=dataset_dir,
        class_to_idx=class_to_idx,
        transforms=get_transform(train=False)
    )

    # For evaluation, it's common to use the entire validation set
    data_loader_val = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=True
    )

    evaluate(model, data_loader_val, device)

if __name__ == '__main__':
    main()
