import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

class DigitDataset(Dataset):
    def __init__(self, images, labels, image_size=48, is_training=True):
        self.images = images
        self.labels = labels
        self.image_size = image_size
        self.is_training = is_training
        self.images_are_pil = isinstance(images[0], Image.Image) if len(images) > 0 else False
        
        # Training transforms with augmentation
        if self.images_are_pil:
            self.train_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=(-30, 30)),
                    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
                    transforms.RandomAffine(degrees=0, shear=[-10, 10, -10, 10]),
                ], p=0.6),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
                ], p=0.2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
            ])
            
            # Validation/test transforms without augmentation
            self.val_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=(-30, 30)),
                    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
                    transforms.RandomAffine(degrees=0, shear=[-10, 10, -10, 10]),
                ], p=0.6),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
                ], p=0.2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
            ])
            
            # Validation/test transforms without augmentation
            self.val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.is_training:
            return self.train_transform(self.images[idx]), self.labels[idx]
        else:
            return self.val_transform(self.images[idx]), self.labels[idx]


def load_digit_dataset(data_dir, test_size=0.2, random_state=42):
    """
    Load PNG images from directory structure: DIGIT_*.png
    where DIGIT is 0-12
    """
    images = []
    labels = []
    
    # Find all PNG files in the directory
    png_files = glob.glob(os.path.join(data_dir, "*.png"))
    
    if not png_files:
        raise ValueError(f"No PNG files found in {data_dir}")
    
    print(f"Found {len(png_files)} PNG files")
    
    # Process each file
    for file_path in tqdm(png_files, desc="Loading images"):
        filename = os.path.basename(file_path)
        
        # Extract digit from filename (assumes format: DIGIT_*.png)
        try:
            digit = int(filename.split('_')[0])
            if digit < 0 or digit > 12:
                print(f"Warning: Skipping file {filename} - digit {digit} out of range (0-12)")
                continue
        except (ValueError, IndexError):
            print(f"Warning: Skipping file {filename} - cannot extract digit from filename")
            continue
        
        # Load image
        try:
            image = Image.open(file_path).convert('RGB')
            images.append(image)
            labels.append(digit)
        except Exception as e:
            print(f"Warning: Could not load image {filename}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images were loaded")

    labels = np.array(labels)
    
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"Dataset loaded: {len(train_images)} training, {len(test_images)} test images")
    print(f"Classes distribution: {np.bincount(labels)}")
    
    return train_images, test_images, train_labels, test_labels


class CifarXModel(nn.Module):
    def __init__(self, num_classes=13, lr=0.001, wd=0.0001):
        super(CifarXModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.wd = wd
        
        # Feature extraction layers with batch normalization and dropout
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after conv layers
        # For 48x48 input: 48 -> 24 -> 12 -> 6
        self.feature_size = 128 * 6 * 6
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Initialize weights
        self._initialize_weights()
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params:,} parameters")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def classify(self, input_image):
        """Classify a single image or batch of images"""
        if isinstance(input_image, Image.Image): # Handle PIL Image
            x = self.transform(input_image)
            x = x.unsqueeze(0)  # Add batch dimension
        else:
            x = input_image
            # Add batch dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(0)

        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)
        
        if predictions.size(0) == 1:
            return predictions.item(), probabilities.squeeze().numpy()
        
        return predictions.numpy(), probabilities.numpy()


def train(model, train_loader, test_loader, optimizer, criterion, epochs, device, scheduler=None):
    model.to(device)
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_samples
        
        #valid 
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
                })
        
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        
        if scheduler:
            scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 50)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'val_loss': avg_val_loss,
            }, "best_digit_model.pth")
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    
    return best_accuracy

def main():
    DATA_DIR = "digits" 
    BATCH_SIZE = 32
    IMAGE_SIZE = 48
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading digit dataset...")
    train_images, test_images, train_labels, test_labels = load_digit_dataset(DATA_DIR)
    
    train_dataset = DigitDataset(train_images, train_labels, image_size=IMAGE_SIZE, is_training=True)
    test_dataset = DigitDataset(test_images, test_labels, image_size=IMAGE_SIZE, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = CifarXModel(num_classes=13, lr=LEARNING_RATE, wd=WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("Starting training...")
    best_accuracy = train(model, train_loader, test_loader, optimizer, criterion, EPOCHS, device, scheduler)
    
    print(f"\nTraining completed! Best validation accuracy: {best_accuracy:.2f}%")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_accuracy': best_accuracy,
    }, "final_digit_model.pth")
    
    print("Model saved as 'final_digit_model.pth'")


def test_single_image(model, image_path, device):
    """Test the model on a single image"""
    image = Image.open(image_path)
    prediction, probabilities = model.classify(image)
    
    print(f"Image: {image_path}")
    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {probabilities[prediction]:.3f}")
    print(f"Top 3 predictions:")
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    for i, idx in enumerate(top_3_indices):
        print(f"  {i+1}. Digit {idx}: {probabilities[idx]:.3f}")


if __name__ == "__main__":
    print("[DIGIT CLASSIFICATION]")
    main()
    
    ### INFERENCE """"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_trained_model("best_digit_model.pth", device)
    # test_single_image(model, "path/to/test/image.png", device)