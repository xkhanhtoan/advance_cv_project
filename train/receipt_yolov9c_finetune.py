from ultralytics import YOLO


def main():
    # Thử sức với YOLOv9c (Compact)
    model = YOLO('yolov9c.pt')

    results = model.train(
        data='/workspace/advance_cv_project/Datasets/RECEIPT_YOLO_TRAIN/data.yaml',

        # --- TRAINING DYNAMICS ---
        epochs=150,  # Train lâu hơn chút vì dùng cos_lr
        patience=50,  # Kiên nhẫn hơn
        batch=8,
        imgsz=1024,
        device=0,

        # --- OPTIMIZER & SCHEDULER ---
        optimizer='AdamW',
        lr0=0.001,  # Khởi điểm thấp
        lrf=0.01,  # Learning rate cuối cùng (Final LR)
        cos_lr=True,  # SỬ DỤNG COSINE SCHEDULER (Chiến lược 2)

        # --- LOSS TUNING (Chiến lược 1) ---
        box=12.0,  # Quan trọng hóa việc vẽ box chuẩn
        cls=0.5,  # Giảm nhẹ phân loại (mặc định 0.5 cũng ok vì v9c tốt rồi)

        # --- AUGMENTATION ---
        rect=True,
        fliplr=0.0,  # Không lật chữ
        degrees=10.0,
        mosaic=1.0,
        close_mosaic=30,  # Tắt mosaic sớm hơn để fine-tune ảnh thật (Chiến lược 4)

        name='receipt_yolov9c_deep_finetune',
        exist_ok=True
    )


if __name__ == '__main__':
    main()