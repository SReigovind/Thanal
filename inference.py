import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Import the architecture from our other file
from model import AttentionUNet

class VNIREngine:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        print(f"⚙️ Initializing VNIR Engine on {self.device}...")
        
        # Initialize Architecture
        self.model = AttentionUNet().to(self.device)
        
        # Load Weights
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Weights loaded from {model_path}")
            except Exception as e:
                print(f"❌ Error loading weights: {e}")
        else:
            print(f"❌ File not found: {model_path}")
            
        self.model.eval()
        
        # Transformations (Must match training)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def predict(self, pil_image):
        """
        Takes a PIL RGB Image, returns a PIL Grayscale VNIR Image
        """
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Post-process
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_array = output_tensor.squeeze().cpu().numpy()
        
        # Convert back to PIL Image
        vnir_image = Image.fromarray((output_array * 255).astype(np.uint8), mode='L')
        return vnir_image