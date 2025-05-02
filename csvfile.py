import os
import pandas as pd

# Define the label folder
label_folder = 'D:/yoga_pose_detection/output/train/labels'

# Create a list to store all rows
all_data = []

# Loop through all .txt files in the label folder
for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_folder, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # Assuming YOLO format: class x_center y_center width height
                    class_id, x_center, y_center, width, height = parts[:5]
                    all_data.append({
                        'filename': filename,
                        'class': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(all_data)
df.to_csv('labels.csv', index=False)

print("Conversion complete. CSV saved as 'labels.csv'")
