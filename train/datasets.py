import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    ])

class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        self.class_names = sorted([d for d in os.listdir(self.root_dir) 
                                if os.path.isdir(os.path.join(self.root_dir, d))])
        self.num_classes = len(self.class_names)
        
        self.samples = []
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        label
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def print_class_distribution(self):
        class_counts = [0] * self.num_classes
        for _, label in self.samples:  # 修改这里，只解包两个值
            class_counts[label] += 1
        print("\nClass Distribution:")
        for class_name, count in zip(self.class_names, class_counts):
            print(f"{class_name}: {count} samples")

def create_datasets(data_root='./animal_dataset'):
    data_root = os.path.abspath(data_root)
    print(f"\nLoading datasets from: {data_root}")
    
    train_path = os.path.join(data_root, 'dataset')
    val_path = os.path.join(data_root, 'eval_dataset')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training directory not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation directory not found at {val_path}")
    
    train_set = AnimalDataset(train_path, get_train_transforms())
    val_set = AnimalDataset(val_path, get_val_transforms())
    
    if train_set.class_names != val_set.class_names:
        raise ValueError("Train and validation sets have different classes!")
    
    print(f"Found {train_set.num_classes} classes: {train_set.class_names}")
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    
    train_set.print_class_distribution()
    val_set.print_class_distribution()
    
    return train_set, val_set