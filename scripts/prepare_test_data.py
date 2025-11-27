import os
import shutil
import urllib.parse
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# 1. Folder chứa dữ liệu Test tải từ Label Studio về (Folder chứa các file .txt)
SOURCE_RAW_LABELS = r'D:\ADMIN\Documents\Classwork\CV\project-16-at-2025-11-26-18-34-b1f82b95\labels'
# Lưu ý: Nếu Label Studio export ra mà ảnh nằm folder khác, hãy trỏ đúng folder ảnh
SOURCE_RAW_IMAGES = r'D:\ADMIN\Documents\Classwork\CV\SROIE2019\test\img'

# 2. Folder đích (Nơi YOLO sẽ đọc)
DEST_DIR = r'D:\ADMIN\Documents\Classwork\CV\Datasets\SROIE_YOLO_TRAIN\test'


def setup_dirs():
    os.makedirs(os.path.join(DEST_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, 'labels'), exist_ok=True)


def clean_filename_and_fix_content():
    setup_dirs()

    # Lấy danh sách file txt
    label_files = glob.glob(os.path.join(SOURCE_RAW_LABELS, "*.txt"))
    print(f"Tìm thấy {len(label_files)} file labels thô. Đang xử lý...")

    count_ok = 0
    count_err = 0

    for long_label_path in label_files:
        if 'classes.txt' in long_label_path: continue

        # 1. XỬ LÝ TÊN FILE (Decode URL)
        filename_only = os.path.basename(long_label_path)  # e81c__ADMIN%5C...txt
        decoded_name = urllib.parse.unquote(filename_only)
        decoded_name = decoded_name.replace('/', '\\')

        # Lấy ID gốc (bỏ đường dẫn rác, bỏ hash)
        base_name = decoded_name.split('\\')[-1]  # X5100.jpg.txt
        if '__' in base_name:
            base_name = base_name.split('__')[-1]

        file_id = base_name.replace('.jpg', '').replace('.png', '').replace('.txt', '')

        real_img_name = file_id + ".jpg"
        real_lbl_name = file_id + ".txt"

        # 2. COPY ẢNH VÀ ĐỔI TÊN
        src_img_path = os.path.join(SOURCE_RAW_IMAGES, real_img_name)
        # Nếu không tìm thấy ảnh theo tên ngắn, thử tìm theo tên dài (do cấu trúc export tùy lúc)
        if not os.path.exists(src_img_path):
            # Thử tìm trong cùng folder với label nếu label studio export chung
            src_img_path = long_label_path.replace('.txt', '')  # tên ảnh gốc thường đi kèm
            if not os.path.exists(src_img_path):
                # Fallback: Quét folder images xem có file nào khớp ID không
                pass  # (Đoạn này giả định cấu trúc chuẩn)

        dst_img_path = os.path.join(DEST_DIR, 'images', real_img_name)
        dst_lbl_path = os.path.join(DEST_DIR, 'labels', real_lbl_name)

        # Chỉ xử lý nếu tìm thấy ảnh gốc
        # Lưu ý: Bạn cần chắc chắn đường dẫn ảnh nguồn đúng
        # Ở đây tôi giả định ảnh nguồn tên là X123.jpg nằm trong SOURCE_RAW_IMAGES
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)

            # 3. ĐỌC VÀ SỬA NỘI DUNG LABEL (Fix Class ID 1 -> 0)
            with open(long_label_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue

                # ÉP CLASS ID VỀ 0
                parts[0] = '0'

                # Kiểm tra xem có cần chia 100 không (nếu > 1)
                coords = [float(p) for p in parts[1:5]]
                if any(c > 1.0 for c in coords):
                    coords = [c / 100.0 for c in coords]
                    parts[1] = f"{coords[0]:.6f}"
                    parts[2] = f"{coords[1]:.6f}"
                    parts[3] = f"{coords[2]:.6f}"
                    parts[4] = f"{coords[3]:.6f}"

                new_lines.append(" ".join(parts) + "\n")

            with open(dst_lbl_path, 'w') as f:
                f.writelines(new_lines)

            count_ok += 1
        else:
            # Rất có thể tên file ảnh bên nguồn cũng bị mã hóa dài loằng ngoằng
            # Trường hợp này copy thẳng file ảnh nguồn tương ứng sang và đổi tên
            # Logic này phức tạp hơn tùy thuộc bản export.
            # Tạm thời báo lỗi để bạn check path.
            print(f"Cảnh báo: Không tìm thấy ảnh gốc cho {file_id} tại {src_img_path}")
            count_err += 1

    print(f"Hoàn tất! Thành công: {count_ok}, Lỗi: {count_err}")
    print(f"Dữ liệu test sạch nằm tại: {DEST_DIR}")


if __name__ == "__main__":
    clean_filename_and_fix_content()