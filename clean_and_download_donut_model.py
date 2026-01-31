"""
clean_and_download_donut_model.py
---------------------------------
Script to clean the models directory and download all required DONUT model files from Hugging Face.
Run this to ensure a fresh, complete, and compatible model setup.
"""


import os
import shutil
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_REPO = "naver-clova-ix/donut-base-finetuned-docvqa"
LOCAL_DIR = "./invoice_engine/models"

# Clean models directory except pytorch_model.bin
if os.path.exists(LOCAL_DIR):
    print(f"Cleaning {LOCAL_DIR} except pytorch_model.bin...")
    for fname in os.listdir(LOCAL_DIR):
        if fname != "pytorch_model.bin":
            fpath = os.path.join(LOCAL_DIR, fname)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
            else:
                os.remove(fpath)
    print("Directory cleaned (except pytorch_model.bin).")
else:
    print(f"{LOCAL_DIR} does not exist. Creating new directory.")
    os.makedirs(LOCAL_DIR, exist_ok=True)

# Download processor and model (skip model weights if already present)
print("Downloading DONUT processor...")
DonutProcessor.from_pretrained(MODEL_REPO, cache_dir=LOCAL_DIR)
if os.path.exists(os.path.join(LOCAL_DIR, "pytorch_model.bin")):
    print("pytorch_model.bin already exists, skipping model weights download.")
    # Download config and other files only
    VisionEncoderDecoderModel.from_pretrained(MODEL_REPO, cache_dir=LOCAL_DIR, ignore_mismatched_sizes=True, local_files_only=False)
else:
    print("Downloading DONUT model (including weights)...")
    VisionEncoderDecoderModel.from_pretrained(MODEL_REPO, cache_dir=LOCAL_DIR)
print("Download complete. Verify all files are present before running the app.")
