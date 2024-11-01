import torch
from torchvision.ops import box_iou
from model import get_model
from dataset import AircraftDataset
from utils import get_transform, class_to_idx
from torch.utils.data import DataLoader

def evaluate(model, data_loader, device):
    model.eval()
    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i in range(len(images)):
                pred_boxes = outputs[i]['boxes'].cpu()
                true_boxes = targets[i]['boxes']

                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue

                iou = box_iou(pred_boxes, true_boxes).numpy()
                total_iou += np.sum(iou)
                total_samples += iou.size

    mean_iou = total_iou / total_samples if total_samples > 0 else 0
    print(f"Mean IoU: {mean_iou:.4f}")

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Number of classes (including background)
    num_classes = len(class_to_idx)

    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load('aircraft_detector.pth'))
    model.to(device)

    # Prepare the validation dataset
    dataset = AircraftDataset(
        dataset_dir='data/dataset',
        class_to_idx=class_to_idx,
        transforms=get_transform(train=False)
    )

    # Assuming you have a separate validation dataset
    data_loader_val = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Evaluate the model
    evaluate(model, data_loader_val, device)
