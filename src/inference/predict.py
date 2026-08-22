import cv2
import numpy as np
import torch

from src.models.classifier import SimpleCNN
from src.models.unet import build_unet

def load_models(classifier_path, segmentation_path, device):
    classifier = SimpleCNN().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    segmentation_model = build_unet(encoder_name="resnet34", in_channels=3, out_channels=1)
    segmentation_model = segmentation_model.to(device)
    segmentation_model.load_state_dict(torch.load(segmentation_path, map_location=device))
    segmentation_model.eval()

    return classifier, segmentation_model


def predict(image_path, classifier, segmentation_model, device,
            image_size=128, classification_threshold=0.5):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (image_size, image_size))

    image_input = image_resized.astype(np.float32) / 255.0
    image_input = np.transpose(image_input, (2, 0, 1))
    image_tensor = torch.tensor(image_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        classifier_output = classifier(image_tensor)
        classifier_prob = torch.sigmoid(classifier_output).item()

    has_tumor = classifier_prob > classification_threshold

    result = {
        "has_tumor": has_tumor,
        "confidence": classifier_prob if has_tumor else 1 - classifier_prob,
        "mask": None,
    }

    if has_tumor:
        with torch.no_grad():
            seg_output = segmentation_model(image_tensor)
            seg_prob = torch.sigmoid(seg_output)
            pred_mask = (seg_prob > 0.5).float().cpu().numpy().squeeze()

        result["mask"] = pred_mask

    return result