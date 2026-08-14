import os
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

def build_dataframe(data_dir: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    patient_folders = [f for f in data_dir.iterdir() if f.is_dir()]

    records = []

    for patient_folder in patient_folders:
        mask_files = [f for f in os.listdir(patient_folder) if f.endswith("_mask.tif")]

        for mask_file in mask_files:
            image_file = mask_file.replace("_mask.tif", ".tif")

            image_path = patient_folder / image_file
            mask_path = patient_folder / mask_file

            records.append({
                "patient_id": patient_folder.name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            })

    df = pd.DataFrame(records)
    return df

def has_tumor(mask_path: str) -> bool:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return np.max(mask) > 0

def prepare_dataset(data_dir: str) -> pd.DataFrame:
    df = build_dataframe(data_dir)
    df["has_tumor"] = df["mask_path"].apply(has_tumor)
    return df