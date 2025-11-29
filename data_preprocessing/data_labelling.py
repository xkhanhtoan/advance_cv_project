from ultralytics import YOLO
import cv2
import os
import json
import uuid
import urllib.parse

# --- CẤU HÌNH ---

# 1. Đường dẫn model best.pt
MODEL_PATH = r'..\train\runs\detect\sroie_yolov9c_finetune\weights\best.pt'

# 2. Thư mục chứa ảnh đầu vào
INPUT_FOLDER = r'D:\ADMIN\Documents\Classwork\advance_cv_project\preprocessed'

# 3. Tên file JSON kết quả
OUTPUT_JSON = 'output_results_labelstudio.json'

# 4. Prefix đường dẫn Local Storage
# (ADMIN\Documents... -> ADMIN%5CDocuments...)
LOCAL_STORAGE_PREFIX = "/data/local-files/?d=ADMIN%5CDocuments%5CClasswork%5Cadvance_cv_project%5Cpreprocessed%5C"

# --- XỬ LÝ ---

def main():
    # Kiểm tra model tồn tại
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy model tại {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Lấy danh sách ảnh
    if not os.path.exists(INPUT_FOLDER):
        print(f"LỖI: Không tìm thấy thư mục ảnh {INPUT_FOLDER}")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Tìm thấy {len(image_files)} ảnh trong {INPUT_FOLDER}. Đang xử lý...")

    ls_tasks = []

    for filename in image_files:
        image_path = os.path.join(INPUT_FOLDER, filename)

        # 1. Đọc ảnh để lấy kích thước gốc
        img = cv2.imread(image_path)
        if img is None: continue
        height, width = img.shape[:2]

        # 2. Dự đoán
        results = model.predict(source=image_path, conf=0.4, verbose=False)

        # 3. Tạo list kết quả
        result_items = []

        for box in results[0].boxes:
            # Tọa độ pixel
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [float(v) for v in coords]

            box_w = float(x2 - x1)
            box_h = float(y2 - y1)

            # Convert sang %
            x_pct = float((x1 / width) * 100)
            y_pct = float((y1 / height) * 100)
            w_pct = float((box_w / width) * 100)
            h_pct = float((box_h / height) * 100)


            region_id = str(uuid.uuid4())[:8]

            item = {
                "id": region_id,
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": ["Text"]
                }
            }
            result_items.append(item)

        # 4. Tạo URL ảnh chuẩn Label Studio
        # Nối Prefix + Tên file (được mã hóa nếu có dấu cách)
        encoded_filename = urllib.parse.quote(filename)
        image_url = LOCAL_STORAGE_PREFIX + encoded_filename

        # 5. Đóng gói JSON
        # Dùng "predictions" để có thể xem và chỉnh sửa trên Label Studio
        task = {
            "data": {
                "image": image_url
            },
            "predictions": [{
                "model_version": "yolov8_v2_best",
                "score": 0.95,
                "result": result_items
            }]
        }
        ls_tasks.append(task)

    # 6. Xuất file
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(ls_tasks, f, ensure_ascii=False, indent=2)

    print(f"\n--- XONG! ---")
    print(f"File JSON đã lưu tại: {os.path.abspath(OUTPUT_JSON)}")
    print(f"Số lượng ảnh đã xử lý: {len(ls_tasks)}")

if __name__ == '__main__':
    main()