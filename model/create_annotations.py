import os
import pandas as pd

data_dir = './data/kaggle'
annotation_file = './data/kaggle_data_labels.csv'

# Define the subdirectories and their corresponding labels
subdirs = {
    'drone': 'drone',
    'drone_far': 'drone_far',
    'noise': 'noise'
}

# Initialize a list to store the annotations
annotations = []

# Iterate through each subdirectory and collect file paths and labels
for subdir, label in subdirs.items():
    subdir_path = os.path.join(data_dir, subdir)
    for filename in os.listdir(subdir_path):
        if filename.endswith('.png'):
            file_path = os.path.join(subdir, filename)
            annotations.append([file_path, label])

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(annotations, columns=['img_path', 'label'])
df.to_csv(annotation_file, index=False)

print(f"Annotation file created: {annotation_file}")