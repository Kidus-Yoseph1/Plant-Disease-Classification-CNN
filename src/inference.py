import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple

def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Loads a trained EfficientNet-B0 model with a custom classifier."""
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    
    # Recreate the classifier to match the trained model
    num_in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=num_in_features,
                        out_features=num_classes,
                        bias=True)
    ).to(device)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def predict(model: torch.nn.Module, 
            image_path: str, 
            class_names: List[str], 
            device: str = "cpu",
            transform: transforms.Compose = None) -> Tuple[str, float]:
    """Predicts the class and probability of a single image."""
    if transform is None:
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        transform = weights.transforms()
        
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        
    return class_names[pred_label], probs[0][pred_label].item()

def predict_directory(model: torch.nn.Module, 
                      dir_path: str, 
                      class_names: List[str], 
                      device: str = "cpu",
                      transform: transforms.Compose = None) -> List[Tuple[str, str, float]]:
    """Predicts classes for all images in a directory."""
    results = []
    for img_name in os.listdir(dir_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dir_path, img_name)
            pred_class, prob = predict(model, img_path, class_names, device, transform)
            results.append((img_name, pred_class, prob))
    return results
