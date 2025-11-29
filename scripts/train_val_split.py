import os
import shutil
import random
import urllib.parse

# --- CẤU HÌNH ---
SOURCE_IMAGES_DIR = r'D:\ADMIN\Documents\Classwork\advance_cv_project\data\Receipt_OCR_2\pre-processed'
SOURCE_LABELS_DIR = r'D:\ADMIN\Documents\Classwork\advance_cv_project\project-5-at-2025-11-29-21-33-56d17241\labels'
DEST_DIR = r'../Datasets/RECEIPT_YOLO_TRAINx'

# Tỷ lệ Train 80%, còn lại 20% là Val
TRAIN_RATIO = 0.8

def setup_dirs():
    # Chỉ tạo thư mục train và val
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_DIR, split, kind), exist_ok=True)

def clean_filename(long_filename):
    decoded_name = urllib.parse.unquote(long_filename)
    decoded_name = decoded_name.replace('/', '\\')
    base_name_with_ext = decoded_name.split('\\')[-1]
    if '__' in base_name_with_ext:
        base_name_with_ext = base_name_with_ext.split('__')[-1]
    file_id = base_name_with_ext.replace('.jpg', '').replace('.png', '').replace('.txt', '')
    return file_id

def copy_data():
    # Xóa thư mục cũ nếu muốn làm lại từ đầu (cẩn thận)
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    setup_dirs()

    label_files = [f for f in os.listdir(SOURCE_LABELS_DIR) if f.endswith('.txt') and f != 'classes.txt']
    random.shuffle(label_files)

    # Chia 80 - 20
    split_idx = int(len(label_files) * TRAIN_RATIO)

    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:] # 20% còn lại

    print(f"Tổng: {len(label_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

    def process_files(files, split_type):
        count_success = 0
        for long_label_file in files:
            file_id = clean_filename(long_label_file)
            real_image_name = file_id + ".jpg"
            real_label_name = file_id + ".txt"

            src_img = os.path.join(SOURCE_IMAGES_DIR, real_image_name)
            src_lbl = os.path.join(SOURCE_LABELS_DIR, long_label_file)

            dst_img = os.path.join(DEST_DIR, split_type, 'images', real_image_name)
            dst_lbl = os.path.join(DEST_DIR, split_type, 'labels', real_label_name)

            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)
                count_success += 1
            else:
                 # Thử tìm png
                src_img_png = src_img.replace(".jpg", ".png")
                if os.path.exists(src_img_png):
                     dst_img = dst_img.replace(".jpg", ".png")
                     shutil.copy(src_img_png, dst_img)
                     shutil.copy(src_lbl, dst_lbl)
                     count_success += 1

        print(f"-> {split_type}: Xong {count_success} file.")

    process_files(train_files, 'train')
    process_files(val_files, 'val')
    print("Hoàn tất!")

if __name__ == "__main__":
    copy_data()