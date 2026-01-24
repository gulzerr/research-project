import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from cnnClassifier import FogResNet50Classifier

class FogDataset(Dataset):
    def __init__(self, dataframe, images_folder, transform=None):
        """
        Args:
            dataframe: pandas DataFrame with columns ['image_id', 'image_path', 'score', 'weak_label']
            images_folder: path to folder containing images
            transform: torchvision transforms
        """
        self.df = dataframe.reset_index(drop=True)
        self.images_folder = images_folder
        self.transform = transform
        
        # Map labels to integers
        self.label_map = {'clear': 0, 'light_fog': 1, 'dense_fog': 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image filename from image_id
        image_filename = row['image_id']
        image_path = os.path.join(self.images_folder, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Get label and normalized score (normalize to 0-1 range for regression head)
        label = self.label_map[row['weak_label']]
        # Normalize score: cap at max 3.0 and divide by 3
        normalized_score = min(float(row['score']), 3.0) / 3.0
        
        return image, label, normalized_score

def get_transforms(is_training=True):
    """Get data augmentation transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def calculate_class_weights(df):
    """Calculate class weights for handling imbalance"""
    label_counts = df['weak_label'].value_counts()
    total = len(df)
    
    weights = {}
    for label in ['clear', 'light_fog', 'dense_fog']:
        if label in label_counts:
            weights[label] = total / (len(label_counts) * label_counts[label])
        else:
            weights[label] = 1.0
    
    # Convert to tensor in order [clear, light_fog, dense_fog]
    weight_tensor = torch.tensor([weights['clear'], weights['light_fog'], weights['dense_fog']], dtype=torch.float32)
    
    print(f"Class distribution:")
    print(label_counts)
    print(f"\nClass weights: {weight_tensor}")
    
    return weight_tensor

def create_weak_labels(df):
    """
    Create weak labels from FADE scores.
    Thresholds: 
    - clear: 0-1
    - light_fog: 1-2
    - dense_fog: >2
    """
    if 'weak_label' in df.columns:
        print("Weak labels already exist in dataframe")
        return df
    
    if 'score' not in df.columns:
        raise ValueError("DataFrame must contain 'score' column")
    
    print("Creating weak labels from FADE scores...")
    print(f"Score range: {df['score'].min():.2f} - {df['score'].max():.2f}")
    
    def score_to_label(score):
        if score < 1.0:
            return 'clear'
        elif score < 2.0:
            return 'light_fog'
        else:
            return 'dense_fog'
    
    df['weak_label'] = df['score'].apply(score_to_label)
    
    print("\nWeak labels created:")
    print(df['weak_label'].value_counts())
    print("\nScore distribution per label:")
    for label in ['clear', 'light_fog', 'dense_fog']:
        if label in df['weak_label'].values:
            scores = df[df['weak_label'] == label]['score']
            print(f"{label}: mean={scores.mean():.2f}, min={scores.min():.2f}, max={scores.max():.2f}")
    
    return df

def train_epoch(model, train_loader, class_criterion, density_criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels, fade_scores in pbar:
        images = images.to(device)
        labels = labels.to(device)
        fade_scores = fade_scores.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        class_logits, density_pred = model(images)
        
        # Calculate losses
        class_loss = class_criterion(class_logits, labels)
        density_loss = density_criterion(density_pred, fade_scores)
        loss = class_loss + 0.5 * density_loss  # Combined loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(class_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, class_criterion, density_criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_densities = []
    all_fade_scores = []
    
    with torch.no_grad():
        for images, labels, fade_scores in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            fade_scores = fade_scores.to(device).float().unsqueeze(1)
            
            class_logits, density_pred = model(images)
            
            class_loss = class_criterion(class_logits, labels)
            density_loss = density_criterion(density_pred, fade_scores)
            loss = class_loss + 0.5 * density_loss
            
            total_loss += loss.item()
            _, predicted = torch.max(class_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions for analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_densities.extend(density_pred.cpu().numpy())
            all_fade_scores.extend(fade_scores.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    # Calculate correlation between predicted density and FADE scores
    correlation = np.corrcoef(np.array(all_densities).flatten(), 
                             np.array(all_fade_scores).flatten())[0, 1]
    
    return avg_loss, accuracy, correlation

def main():
    # Get the project root directory (parent of classifier folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration - Use absolute paths
    PARQUET_FILE = os.path.join(project_root, 'data', 'fade_results_complete.parquet')
    IMAGES_FOLDER = os.path.join(project_root, 'images')
    MODELS_FOLDER = os.path.join(project_root, 'models')
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models folder if it doesn't exist
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    print(f"Models will be saved to: {MODELS_FOLDER}")
    
    print(f"Using device: {DEVICE}")
    
    # Check if files exist
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_FILE}\n"
                              f"Please ensure the file exists at: {os.path.abspath(PARQUET_FILE)}")
    
    if not os.path.exists(IMAGES_FOLDER):
        raise FileNotFoundError(f"Images folder not found: {IMAGES_FOLDER}\n"
                              f"Please ensure the folder exists at: {os.path.abspath(IMAGES_FOLDER)}")
    
    # Load data
    print(f"\nLoading data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show first few rows and statistics
    print("\nFirst few rows:")
    print(df.head())
    print(f"\nScore statistics:")
    print(df['score'].describe())
    
    # Create weak labels based on score thresholds
    df = create_weak_labels(df)
    
    # Check if we have enough data for each class
    min_samples_per_class = df['weak_label'].value_counts().min()
    
    # Handle small datasets
    if len(df) < 10:
        print(f"\nWARNING: Very small dataset ({len(df)} samples)!")
        print("Using all data for both training and testing (no split).")
        train_df = df.copy()
        test_df = df.copy()  # Use same data for testing
    elif min_samples_per_class < 2:
        print(f"\nWARNING: Some classes have fewer than 2 samples!")
        print("Splitting without stratification...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    else:
        # Normal split with stratification (80% train, 20% test)
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                                stratify=df['weak_label'])
        except ValueError as e:
            print(f"\nError splitting data: {e}")
            print("Splitting without stratification...")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"\nTrain samples: {len(train_df)} (80%)")
    print(f"Test samples: {len(test_df)} (20%)")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_df).to(DEVICE)
    
    # Create datasets
    train_dataset = FogDataset(train_df, IMAGES_FOLDER, transform=get_transforms(is_training=True))
    test_dataset = FogDataset(test_df, IMAGES_FOLDER, transform=get_transforms(is_training=False))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Initialize model
    print("\nInitializing ResNet-50 model...")
    model = FogResNet50Classifier(num_classes=3, pretrained=True).to(DEVICE)
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    density_criterion = nn.MSELoss()
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_val_acc = 0
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print('='*50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, class_criterion, 
                                           density_criterion, optimizer, DEVICE)
        
        # Test
        test_loss, test_acc, correlation = validate(model, test_loader, class_criterion, 
                                                  density_criterion, DEVICE)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Density Correlation: {correlation:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model to models folder
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            model_path = os.path.join(MODELS_FOLDER, 'best_fog_resnet50.pth')
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'correlation': correlation,
            }, model_path)
            
            print(f"  ✓ Saved best model to: {model_path}")
            
            # Verify the file was created
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
                print(f"  ✓ Model file verified ({file_size:.2f} MB)")
            else:
                print(f"  ✗ ERROR: Model file was not created!")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best test accuracy: {best_val_acc:.2f}%")
    print(f"Model saved at: {os.path.join(MODELS_FOLDER, 'best_fog_resnet50.pth')}")
    print("="*50)

if __name__ == '__main__':
    main()