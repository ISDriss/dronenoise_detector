# https://www.kaggle.com/datasets/hitheshs/drone-noise-spectrograms?resource=download

import kagglehub

# Download latest version
path = kagglehub.dataset_download("hitheshs/drone-noise-spectrograms")

print("Path to dataset files:", path)