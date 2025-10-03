import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import ResNet50_Weights, resnet50
import numpy as np
import os

#recieve pretrained model
def create_model():
    model = resnet50(weights=ResNet50_Weights.SENTINEL1_ALL_MOCO)

    original_conv = model.conv1
    model.conv1 = nn.Conv2d(4, original_conv.out_channels,
                            kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        model.conv1.weight[:, :2, :, :] = original_conv.weight
        model.conv1.weight[:, 2:, :, :] = original_conv.weight

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


_MODEL = None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_filename='quake_sar_classifier_FINAL.pth'):
    global _MODEL
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_filename)
    # Load the model
    checkpoint = torch.load(model_path, map_location=_DEVICE)
    _MODEL = create_model()
    _MODEL.load_state_dict(checkpoint['model_state_dict'])
    _MODEL.to(_DEVICE)
    _MODEL.eval()

load_model()#load model once

#preprocess input sar image
def preprocess(sar_image):
    if isinstance(sar_image, np.ndarray):
        sar_image = torch.from_numpy(sar_image).float()

    if sar_image.dim() == 3:
        sar_image = sar_image.unsqueeze(0)  # (4, H, W) → (1, 4, H, W)

    # Resize to 224x224 if needed
    if sar_image.shape[2] != 224 or sar_image.shape[3] != 224:
        sar_image = F.interpolate(
            sar_image,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

    return sar_image.to(_DEVICE)

#main prediction function
def predict_damage(sar_image):
    sar_tensor = preprocess(sar_image)

    with torch.no_grad():
        output = _MODEL(sar_tensor)
        probs = F.softmax(output, dim=1)
        prediction_id = torch.argmax(probs, dim=1).item()

    return ['No Damage', 'Damage'][prediction_id]