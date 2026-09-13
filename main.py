"""
Command-line tool for running brain tumor detection & segmentation
on a single MRI image, without needing the API server or browser.

Usage:
    python main.py --image path/to/scan.tif
    python main.py --image path/to/scan.tif --save-mask outputs/predictions/mask.png
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.inference.predict import load_models, predict


CLASSIFIER_PATH = "models/classification/weights/classifier_weighted_best.pt"
SEGMENTATION_PATH = "models/segmentation/weights/unet_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run brain tumor detection and segmentation on an MRI image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the MRI image file (.tif, .png, .jpg, etc.)",
    )
    parser.add_argument(
        "--save-mask",
        default=None,
        help="Optional path to save the predicted tumor mask overlay as a PNG file.",
    )
    return parser.parse_args()


def save_mask_overlay(mask, save_path):
    import numpy as np

    height, width = mask.shape
    overlay = np.zeros((height, width, 4), dtype="uint8")
    overlay[mask > 0] = [255, 70, 70, 200]

    image = Image.fromarray(overlay, mode="RGBA")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    classifier, segmentation_model = load_models(
        classifier_path=CLASSIFIER_PATH,
        segmentation_path=SEGMENTATION_PATH,
        device=device,
    )

    print(f"Running prediction on: {image_path}")
    result = predict(
        image_path=str(image_path),
        classifier=classifier,
        segmentation_model=segmentation_model,
        device=device,
    )

    print("\n--- Prediction Result ---")
    print(f"Tumor detected: {result['has_tumor']}")
    print(f"Confidence: {result['confidence']:.2%}")

    if result["has_tumor"] and result["mask"] is not None:
        tumor_pixel_count = int(result["mask"].sum())
        total_pixels = result["mask"].size
        print(f"Tumor area: {tumor_pixel_count}/{total_pixels} pixels "
              f"({tumor_pixel_count / total_pixels:.2%} of the image)")

        if args.save_mask:
            save_mask_overlay(result["mask"], args.save_mask)
            print(f"Mask overlay saved to: {args.save_mask}")
    else:
        print("No tumor mask to display.")


if __name__ == "__main__":
    main()