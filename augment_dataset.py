import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import albumentations as A
from pathlib import Path

# 1. Конфигурация
INPUT_DATASET = "paxray_yolo_seg"
OUTPUT_DATASET = "paxray_yolo_seg_augmented"
APPLY_TO_TRAIN = True  
APPLY_TO_VAL = False   

# 2. Создаем аугментации
train_aug = A.Compose([
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.ISONoise(color_shift=(0.01, 0.05)),
    ], p=0.4),
])

val_aug = A.Compose([
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True)
])

# 3. Создаем структуру папок
Path(f"{OUTPUT_DATASET}/images/train").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DATASET}/images/val").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DATASET}/labels/train").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DATASET}/labels/val").mkdir(parents=True, exist_ok=True)

# 4. Функция для обработки изображений
def process_images(input_dir, output_dir, subset, augmentations):
    images_dir = Path(f"{input_dir}/images/{subset}")
    labels_dir = Path(f"{input_dir}/labels/{subset}")
    
    if not images_dir.exists():
        print(f"Папка {images_dir} не найдена!")
        return
    
    images = list(images_dir.glob("*.*"))
    print(f"\n🔧 Обработка {subset} ({len(images)} изображений)")
    
    for img_path in tqdm(images, desc=f"Аугментация {subset}"):
        image = cv2.imread(str(img_path))
        if image is not None:
            augmented = augmentations(image=image)
            cv2.imwrite(f"{output_dir}/images/{subset}/{img_path.name}", augmented['image'])
        
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, f"{output_dir}/labels/{subset}/{label_path.name}")

# 5. Главная функция
def main():  
    if APPLY_TO_TRAIN:
        process_images(INPUT_DATASET, OUTPUT_DATASET, "train", train_aug)
    
    process_images(INPUT_DATASET, OUTPUT_DATASET, "val", 
                  val_aug if APPLY_TO_VAL else A.Compose([]))

if __name__ == "__main__":
    main()