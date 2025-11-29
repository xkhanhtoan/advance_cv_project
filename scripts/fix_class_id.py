import os
import glob

# ĐƯỜNG DẪN DATASET
DATASET_DIR = '/workspace/Datasets/SROIE_YOLO_TRAIN'

def fix_class_ids():
    # Tìm tất cả file .txt
    label_files = glob.glob(os.path.join(DATASET_DIR, '**', '*.txt'), recursive=True)
    
    print(f"Tìm thấy {len(label_files)} file. Đang chuyển Class ID về 0...")
    
    count = 0
    for file_path in label_files:
        if file_path.endswith('classes.txt') or 'cache' in file_path:
            continue
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # --- SỬA LỖI Ở ĐÂY ---
            # Nếu Class ID khác '0', ép nó về '0'
            if parts[0] != '0':
                parts[0] = '0' # Ép về 0
                modified = True
            
            # Ghép lại dòng mới
            new_line = " ".join(parts) + "\n"
            new_lines.append(new_line)
            
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            count += 1

    print(f"Xong! Đã sửa {count} file.")
    
    # Xóa cache lần nữa cho chắc
    os.system(f"rm -rf {DATASET_DIR}/train/labels.cache")
    os.system(f"rm -rf {DATASET_DIR}/val/labels.cache")
    print("Đã xóa cache cũ.")

if __name__ == "__main__":
    fix_class_ids()