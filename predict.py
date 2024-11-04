import torch
from torchvision import transforms
from model import get_model
from utils import idx_to_class
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def predict(model, image_path, device, threshold=0.5):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(image).to(device)
    
    with torch.no_grad():
        outputs = model([img])

    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

    # Process outputs
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    print(f"Total detections: {len(boxes)}")
    print(f"Scores: {scores}")

    # Filter out low confidence predictions
    high_conf_indices = scores > threshold
    boxes = boxes[high_conf_indices]
    labels = labels[high_conf_indices]
    scores = scores[high_conf_indices]

    print(f"Detections above threshold ({threshold}): {len(boxes)}")

    if len(boxes) == 0:
        print("No detections above the threshold.")
    else:
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='red', width=2)
            class_name = idx_to_class.get(label.item(), "Unknown")
            draw.text((xmin, ymin - 10), f"{class_name}: {score:.2f}", fill='red', font=font)

    plt.figure(figsize=(12,8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Number of classes (including background)
    num_classes = len(idx_to_class)

    # Load the trained model
    from model import get_model
    model = get_model(num_classes)
    model.load_state_dict(torch.load('aircraft_detector.pth'))
    model.to(device)

    # Path to the image you want to test
    image_path = 'f18.jpg'

    # Perform prediction
    predict(model, image_path, device, threshold=0.1)
