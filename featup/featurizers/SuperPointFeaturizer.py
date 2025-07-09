import torch
from torch import nn
from .SuperPoint.SuperPoint import SuperPoint

class SuperPointFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SuperPoint()
        self.model.eval()
        
    def forward(self, img):
        """
        Extract SuperPoint features from input image
        
        Args:
            img: Input image tensor of shape (B, C, H, W)
            
        Returns:
            features: Feature tensor of shape (B, 256, H//8, W//8)
        """
        # SuperPoint expects RGB input and handles grayscale conversion internally
        features = self.model(img)
        return features
