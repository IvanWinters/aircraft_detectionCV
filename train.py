import os
import torch
from torch.utils.data import DataLoader
from model import get_model
from dataset import AircraftDataset
from utils import get_transform, class_to_idx
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Number of classes including background
    num_classes = len(class_to_idx)

    # Get the model
    model = get_model(num_classes)
    model.to(device)

    # Dataset directory
    dataset_dir = 'data/dataset'  # Adjusted path

    # Prepare the dataset and dataloaders
    dataset = AircraftDataset(
        dataset_dir=dataset_dir,
        class_to_idx=class_to_idx,
        transforms=get_transform(train=True)
    )

    # Split dataset into training and validation sets
    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.8 * len(dataset))
    dataset_train = torch.utils.data.Subset(dataset, indices[:split])
    dataset_val = torch.utils.data.Subset(dataset, indices[split:])

    # Data loaders
    data_loader_train = DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=collate_fn
    )

    data_loader_val = DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    # Optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in data_loader_train:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader_train)}], Loss: {losses.item():.4f}")
            i += 1

        # Update the learning rate
        lr_scheduler.step()

        # Optionally evaluate the model here

    # Save the trained model
    torch.save(model.state_dict(), 'aircraft_detector.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()