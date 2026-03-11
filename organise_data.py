import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# organize files into subdirectories based on leading digit
for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            digit = file.split('_')[0]
            if not digit.isdigit():
                continue
            os.makedirs(os.path.join(DATA_DIR, digit), exist_ok=True)
            src_path = os.path.join(root, file)
            dst_path = os.path.join(DATA_DIR, digit, file)
            os.rename(src_path, dst_path)

# remove empty directories
for dir in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir)
    if os.path.isdir(dir_path) and not os.listdir(dir_path):
        os.rmdir(dir_path)

print("Files have been organized into subdirectories based on their leading digit.")