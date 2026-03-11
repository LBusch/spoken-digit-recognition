import pandas as pd
import os

DATA_PATH = "data"
CSV_PATH = "data/audioMNIST.csv"

file_paths = []
labels = []
data_dict = {"file_path": file_paths, "label": labels}

# go through each subdirectory in data directory and get file paths and labels
dirs = os.listdir(DATA_PATH)
for dir in dirs:
    if not os.path.isdir(os.path.join(DATA_PATH, dir)):
        continue
    for file in os.listdir(os.path.join(DATA_PATH, dir)):
        if file.endswith(".wav"):
            file_paths.append(os.path.join(DATA_PATH, dir, file))
            labels.append(dir)

# save file paths and labels to csv
df = pd.DataFrame(data_dict)
df.to_csv(CSV_PATH, index=False)

