from ultralytics import YOLO

def main():
    # 1. ĐỔI MODEL: Dùng bản Small (s) hoặc Medium (m) thay vì Nano (n)
    # Khuyên dùng: yolov8s.pt (cân bằng) hoặc yolov8m.pt (chính xác cao nhất)
    model = YOLO('../yolov8m.pt')

    # 2. HUẤN LUYỆN
    results = model.train(
        data='/workspace/Datasets/SROIE_YOLO_TRAIN/data.yaml',
        
        # --- CẤU HÌNH NÂNG CAO ---
        epochs=100,           # Tăng số vòng lặp để model học kỹ hơn
        imgsz=1024,           # QUAN TRỌNG: Tăng độ phân giải để nhìn rõ chữ nhỏ
        rect=True,            # Giữ tỉ lệ ảnh chữ nhật (giảm viền đen)
        
        batch=8,              # Giảm batch xuống vì imgsz tăng (tránh tràn RAM GPU)
        device=0,
        
        # --- AUGMENTATION (Biến đổi ảnh) ---
        fliplr=0.0,           # Tuyệt đối không lật ngang chữ
        degrees=10.0,         # Cho phép xoay ảnh +/- 10 độ (hóa đơn nghiêng)
        shear=2.5,            # Cho phép méo ảnh nhẹ (chụp xéo)
        mosaic=1.0,           # Bật mosaic ban đầu...
        close_mosaic=20,      # ...nhưng TẮT mosaic ở 20 epoch cuối để tinh chỉnh độ chính xác
        
        # --- TỐI ƯU HÓA ---
        patience=30,          # Nếu 30 epoch không khá hơn thì dừng
        optimizer='AdamW',    # AdamW thường hội tụ tốt hơn SGD cho dataset nhỏ/vừa
        lr0=0.001,            # Learning rate khởi điểm (mặc định 0.01 hơi lớn cho AdamW)
        
        name='sroie_best_v2', # Lưu vào folder mới
        exist_ok=True
    )

if __name__ == '__main__':
    main()