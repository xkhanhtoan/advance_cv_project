import json
import os
import cv2

# --- CẤU HÌNH ---
IMG_FOLDER = r'D:\ADMIN\Documents\Classwork\CV\SROIE2019\test\img'       # Đường dẫn chứa ảnh trên máy bạn
TXT_FOLDER = r'D:\ADMIN\Documents\Classwork\CV\SROIE2019\test\box' # Đường dẫn chứa file txt

# --- SỬA ĐỔI QUAN TRỌNG Ở ĐÂY ---
# Copy chính xác phần đường dẫn đứng trước tên file trong snippet bạn gửi
# Lưu ý: Không bao gồm tên file (ví dụ X5100...jpg), chỉ lấy phần thư mục
STORAGE_PREFIX = "/data/local-files/?d=ADMIN%5CDocuments%5CClasswork%5CCV%5CSROIE2019%5Ctest%5Cimg%5C"

OUTPUT_JSON = 'sroie_test.json'

data_list = []

print("Đang xử lý...")

for filename in os.listdir(IMG_FOLDER):
    if not filename.endswith(('.jpg', '.png', '.jpeg')):
        continue

    # ... (Giữ nguyên phần đọc ảnh và txt như code cũ) ...
    img_path = os.path.join(IMG_FOLDER, filename)
    img = cv2.imread(img_path)
    if img is None: continue
    height, width, _ = img.shape

    txt_name = os.path.splitext(filename)[0] + '.txt'
    txt_path = os.path.join(TXT_FOLDER, txt_name)
    if not os.path.exists(txt_path): continue

    # TẠO TASK
    task = {
        "data": {
            # Ghép Prefix mã hóa + tên file gốc
            "image": STORAGE_PREFIX + filename
        },
        "annotations": [{
            "result": []
        }]
    }

    # ... (Giữ nguyên phần vòng lặp đọc file txt và tính toán tọa độ) ...
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) < 9: continue

            coords = [int(p) for p in parts[:8]]
            text = ",".join(parts[8:])

            x_vals = coords[0::2]
            y_vals = coords[1::2]
            x_min, y_min = min(x_vals), min(y_vals)
            w_px = max(x_vals) - x_min
            h_px = max(y_vals) - y_min

            x_pct = (x_min / width) * 100
            y_pct = (y_min / height) * 100
            w_pct = (w_px / width) * 100
            h_pct = (h_px / height) * 100

            import uuid
            region_id = str(uuid.uuid4())[:8]

            # BBOX
            bbox_result = {
                "id": region_id,
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct,
                    "rotation": 0, "rectanglelabels": ["Text"]
                }
            }
            # TEXT
            text_result = {
                "id": region_id,
                "from_name": "transcription",
                "to_name": "image",
                "type": "textarea",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct,
                    "rotation": 0, "text": [text]
                }
            }
            task["annotations"][0]["result"].extend([bbox_result, text_result])

    data_list.append(task)

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"Xong! Hãy import file {OUTPUT_JSON}")