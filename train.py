from ultralytics import YOLO

def main():
    model = YOLO("runs/train/vertebrae_final/weights/best.pt")
    
    results = model.train(
        data="C:/yolo_projects/paxray_yolo_seg_augmented/dataset.yaml",
        epochs=30,
        lr0=0.0001,
        batch=8,
        imgsz=640,
        augment=True,
        workers=0,
        device='0',
        project="runs/two_stage_fixed",
        name="stage2_fixed",
        exist_ok=True
    )
    
    print("✅ Fine-tuning завершен!")

if __name__ == '__main__':
    main()