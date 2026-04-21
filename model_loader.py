import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class DrowsinessModelLoader:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        
        # Standard MobileNetV2 transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        model = models.mobilenet_v2()
        # The model appears to have 2 classes: Open and Closed eyes
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # Remove "module." prefix from DataParallel
            new_k = k.replace('module.', '')
            
            # Map classifier.1.1 to classifier.1
            if "classifier.1.1" in new_k:
                new_k = new_k.replace("classifier.1.1", "classifier.1")
                
            new_state_dict[new_k] = v
            
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict_eye_state(self, eye_image):
        """
        Expects a PIL image of an eye.
        Returns: 0 for Closed, 1 for Open (assuming standard ordering, will adjust if needed)
        """
        if eye_image is None or eye_image.size == 0:
            return 1 # Default to open if detection fails
            
        img_tensor = self.transform(eye_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()
