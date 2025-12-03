from ultralytics import YOLO

MODELS_TO_TRAIN = [
    {'name': 'yolov8n', 'file': 'yolov8n.pt', 'desc': 'Nano_Full_Power'},
    {'name': 'yolov8m', 'file': 'yolov8m.pt', 'desc': 'Medium_Rematch_v9'}
]

def main():
    for model_info in MODELS_TO_TRAIN:
        print(f"\n{'='*40}")
        print(f"BẮT ĐẦU TRAIN: {model_info['name'].upper()} ({model_info['desc']})")
        print(f"{'='*40}\n")
        
        # Load model
        model = YOLO(model_info['file'])

        # Sử dụng đúng bộ tham số "SOTA" đã dùng cho v9c
        results = model.train(
            data='/workspace/advance_cv_project/Datasets/SROIE_YOLO_TRAIN/data.yaml',
            
            # --- CẤU HÌNH FAIR PLAY ---
            epochs=150,           # Ngang bằng v9c
            patience=50,
            batch=8,              # v8m và v9c nặng ngang nhau, batch 8 là an toàn
            imgsz=1024,           # Sân chơi công bằng về độ phân giải
            device=0,
            
            # --- OPTIMIZER & SCHEDULER ---
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,          # Cosine Scheduler
            
            # --- LOSS TUNING ---
            box=12.0,             # Trọng số box cao
            cls=0.5,
            
            # --- AUGMENTATION ---
            rect=True,
            fliplr=0.0,
            degrees=10.0,
            mosaic=1.0,
            close_mosaic=30,      # Tắt mosaic sớm
            
            # Lưu tên folder riêng để dễ so sánh
            name=f"sroie_{model_info['name']}_finetune",
            exist_ok=True
        )
        
        print(f"Xong {model_info['name']}. Nghỉ 10s trước khi qua model tiếp theo...")
        import time
        time.sleep(10)

if __name__ == '__main__':
    main()