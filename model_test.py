from ultralytics import YOLO
import cv2, numpy as np, os
from pathlib import Path

model = YOLO(r"runs/train/vertebrae_final/weights/best.pt")

# Папки
input_dir = "test_images"
output_dir = "test_results"

os.makedirs(output_dir, exist_ok=True)

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = [f for f in os.listdir(input_dir) 
               if Path(f).suffix.lower() in image_extensions]

for filename in image_files:
    try:
        img_path = os.path.join(input_dir, filename)
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        results = model(img_rgb, conf=0.261, iou=0.45, max_det=20)[0]
        
        if results.masks:
            # Фильтрация по классам
            class_best = {}
            for mask, box in zip(results.masks.data, results.boxes):
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                if cid not in class_best or conf > class_best[cid][1]:
                    class_best[cid] = (mask.cpu().numpy(), conf, model.names[cid])
            
            # Визуализация
            output = img_rgb.copy()
            colors = [(255,0,0), (255,128,0), (255,255,0), (128,255,0),
                     (0,255,0), (0,255,128), (0,255,255), (0,128,255),
                     (0,0,255), (128,0,255), (255,0,255), (255,0,128)]
            
            for cid, (mask, conf, name) in sorted(class_best.items()):
                m = cv2.resize(mask, img_rgb.shape[:2][::-1])
                mask_area = m > 0.5
                output[mask_area] = output[mask_area] * 0.7 + np.array(colors[cid%12]) * 0.3
                
                y, x = np.where(mask_area)
                if len(x) > 0:
                    cx, cy = int(np.mean(x)), int(np.mean(y))
                    cv2.putText(output, name, (cx-8, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
                    cv2.putText(output, name, (cx-8, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            
            output_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            
        else:
            output_path = os.path.join(output_dir, f"no_detection_{filename}")
            cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    except:
        print(f"Ошибка в {filename}")
        continue