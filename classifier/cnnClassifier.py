import torch.nn as nn
import torchvision.models as models

class FogResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get feature dimension (2048 for ResNet-50)
        num_features = self.backbone.fc.in_features
        
        # Replace final layer with custom heads
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Classification head (clear/light_fog/dense_fog)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Density regression head (aligned with FADE scores)
        self.density_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output 0-1 fog density
        )
    
    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        density_pred = self.density_head(features)
        return class_logits, density_pred