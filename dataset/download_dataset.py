#Import data

import os
import zipfile

# Create data2 directory if it doesn't exist
os.makedirs('./data2', exist_ok=True)

# Download the dataset (the link appears to work without API keys for this dataset/environment)
!curl -L -o ./data2/amazon-ml-challenge-2025.zip https://www.kaggle.com/api/v1/datasets/download/raghavdharwal/amazon-ml-challenge-2025

# Unzip the file
print("Extracting dataset...")
with zipfile.ZipFile('./data2/amazon-ml-challenge-2025.zip', 'r') as zip_ref:
    zip_ref.extractall('./data2')
print("Done!")