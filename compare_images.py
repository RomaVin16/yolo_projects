import cv2
import matplotlib.pyplot as plt
import os

original_dir = "C:/yolo_projects/paxray_yolo_seg/images/train"
augmented_dir = "C:/yolo_projects/paxray_yolo_seg_augmented/images/train"

original_files = sorted(os.listdir(original_dir))
augmented_files = sorted(os.listdir(augmented_dir))

file_index = 5

if file_index < len(original_files):
    original_path = os.path.join(original_dir, original_files[file_index])
    augmented_path = os.path.join(augmented_dir, original_files[file_index])

original = cv2.imread(original_path)
augmented = cv2.imread(augmented_path)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"До аугментации\n{original_files[file_index]}\n(индекс: {file_index})")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
axes[1].set_title("После аугментации")
axes[1].axis('off')

plt.show()