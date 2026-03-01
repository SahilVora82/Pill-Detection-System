import pandas as pd

df = pd.read_csv('table.csv')

# Count unique drug names
num_classes = df['name'].nunique()
print(f"Number of unique drug names: {num_classes}")

# Top 10 most common drug names
top_classes = df['name'].value_counts().head(10)
print("\nTop 10 most common drug names:")
print(top_classes)

import os

# Path to the folder where you copied the images
image_root = 'image/images/gallery'

# Count all .jpg files recursively
image_count = 0
for root, dirs, files in os.walk(image_root):
    image_count += len([f for f in files if f.lower().endswith('.jpg')])

print(f"Total number of pill images: {image_count}")
