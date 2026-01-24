import torch
from PIL import Image
from torchvision import transforms
from cnnClassifier import FogResNet50Classifier
import argparse

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = FogResNet50Classifier(num_classes=3, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def predict_image(model, image_path, device):
    """Predict fog class and density for a single image"""
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        class_logits, density_pred = model(image_tensor)
        probabilities = torch.softmax(class_logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        density = density_pred.item()
    
    # Map class to label
    class_names = ['clear', 'light_fog', 'dense_fog']
    predicted_label = class_names[predicted_class]
    
    # Convert normalized density back to FADE score (0-3 range)
    fade_score = density * 3.0
    
    return {
        'predicted_class': predicted_label,
        'confidence': confidence,
        'fog_density': density,
        'fade_score': fade_score,
        'probabilities': {
            'clear': probabilities[0, 0].item(),
            'light_fog': probabilities[0, 1].item(),
            'dense_fog': probabilities[0, 2].item()
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best_fog_resnet50.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    
    print(f"Predicting for {args.image}...")
    result = predict_image(model, args.image, device)
    
    print(f"\nPrediction:")
    print(f"  Class: {result['predicted_class'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Normalized Density: {result['fog_density']:.3f}")
    print(f"  FADE Score: {result['fade_score']:.3f}")
    print(f"\nProbabilities:")
    for label, prob in result['probabilities'].items():
        bar = '█' * int(prob * 40)
        print(f"  {label:12s}: {prob:6.2%} {bar}")