from torchvision import transforms

def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        # Add more augmentations if needed
    return transforms.Compose(transforms_list)

# Class names including background
classes = [
    '__background__', 'A10', 'A400M', 'AG600', 'AH64', 'AV8B', 'An124',
    'An22', 'An225', 'An72', 'B1', 'B2', 'B21', 'B52', 'Be200', 'C130',
    'C17', 'C2', 'C390', 'C5', 'CH47', 'CL415', 'E2', 'E7', 'EF2000',
    'F117', 'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'F/A18', 'H6',
    'J10', 'J20', 'JAS39', 'JF17', 'JH7', 'KC135', 'KF21', 'KJ600',
    'Ka27', 'Ka52', 'MQ9', 'Mi24', 'Mi26', 'Mi28', 'Mig29', 'Mig31',
    'Mirage2000', 'P3', 'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25', 'Su34',
    'Su57', 'TB001', 'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2',
    'UH60', 'US2', 'V22', 'Vulcan', 'WZ7', 'XB70', 'Y20', 'YF23', 'Z19'
]

# Create a mapping from class name to index
class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

# Reverse mapping from index to class name
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
