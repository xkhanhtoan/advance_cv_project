import cv2
import numpy as np
import os
import math
from datetime import datetime

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
INPUT_FOLDER = r'D:\ADMIN\Documents\Classwork\advance_cv_project\data\Receipt_OCR_0\raw'
OUTPUT_FOLDER = r'D:\ADMIN\Documents\Classwork\advance_cv_project\data\Receipt_OCR_0\pre-processing'
LOG_FILE = r'processing_log.txt'

# ============================================================
# HÀM 1: CẮT ẢNH
# ============================================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

# ============================================================
# HÀM 2: XOAY ẢNH DỰA TRÊN DÒNG CHỮ
# ============================================================
def deskew_image(image):
    # 1. Chuyển xám và tìm biên cạnh (Canny)
    # Nếu ảnh đang là BGR thì chuyển, nếu Gray rồi thì thôi
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Đảo ngược màu (chữ trắng nền đen) để Hough hoạt động tốt hơn
    # Dùng Otsu để tách chữ ra khỏi nền giấy
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 2. Dùng HoughLinesP để tìm các đoạn thẳng
    # minLineLength: Độ dài tối thiểu (chiều rộng ảnh / 2) -> Đảm bảo chỉ bắt dòng chữ dài
    # maxLineGap: Khoảng cách chấp nhận được giữa các từ (để nối các từ thành 1 dòng)
    h, w = image.shape[:2]
    min_line_len = w // 2

    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=200,
                           minLineLength=min_line_len, maxLineGap=20)

    if lines is None:
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=100,
                               minLineLength=w // 4, maxLineGap=20)

    angle = 0.0

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Tính góc: atan2(dy, dx)
            # Đổi sang độ (degree)
            deg = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Chỉ lấy các dòng nằm ngang (nghiêng trong khoảng -45 đến 45 độ)
            # Loại bỏ các đường kẻ dọc của bảng biểu
            if -45 < deg < 45:
                angles.append(deg)

        # 3. Tính trung vị (Median) để loại bỏ nhiễu
        if len(angles) > 0:
            angle = np.median(angles)
            print(f" -> Phát hiện góc nghiêng: {angle:.2f} độ")
        else:
            print(" -> Không tìm thấy dòng ngang nào, giữ nguyên.")
    else:
        print(" -> Không tìm thấy đường thẳng (Hough), giữ nguyên.")

    # 4. Xoay ảnh
    if abs(angle) > 0.1: # Chỉ xoay nếu nghiêng đáng kể
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # borderValue=(255,255,255): Điền nền trắng vào phần hở ra khi xoay
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated

    return image

# ============================================================
# HÀM XỬ LÝ 1 ẢNH
# ============================================================
def process_single_image(img_path, save_dir):
    try:
        # 1. Đọc & Resize
        image = cv2.imread(img_path)
        if image is None: return False, "Lỗi đọc file"
        orig = image.copy()

        ratio = image.shape[0] / 500.0
        image = cv2.resize(image, (int(image.shape[1]/ratio), 500))

        # 2. Tìm Contour (Cắt giấy)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        warped = None

        # Nếu tìm thấy contour giấy -> Cắt
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                warped = four_point_transform(orig, box.reshape(4, 2) * ratio)

        # Nếu không tìm thấy contour (hoặc quá nhỏ), dùng nguyên ảnh gốc để xử lý tiếp
        if warped is None:
            print(" -> Cảnh báo: Không cắt được giấy, dùng ảnh gốc.")
            warped = orig

        # ============================================================
        # 3. XOAY THẲNG (DESKEW)
        # ============================================================
        # Ảnh warped lúc này có thể bị nghiêng chữ, ta xoay nó lại
        deskewed = deskew_image(warped)

        # ============================================================
        # 4. XỬ LÝ OCR (NEUTRAL)
        # ============================================================
        if len(deskewed.shape) == 3:
            deskewed_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        else:
            deskewed_gray = deskewed

        deskewed_gray = cv2.convertScaleAbs(deskewed_gray, alpha=1.3, beta=0)
        deskewed_gray = cv2.GaussianBlur(deskewed_gray, (3, 3), 0)

        warped_bin = cv2.adaptiveThreshold(deskewed_gray, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 9)

        # 5. Lưu ảnh
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        save_path_ocr = os.path.join(save_dir, f"{name}_ocr.jpg")
        cv2.imwrite(save_path_ocr, warped_bin)

        return True, "Thành công"

    except Exception as e:
        return False, f"Lỗi ngoại lệ: {str(e)}"

# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(INPUT_FOLDER): print("Lỗi thư mục input"); return
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_ext]
    print(f"--- BẮT ĐẦU XỬ LÝ {len(files)} ẢNH ---")

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        for i, filename in enumerate(files):
            print(f"[{i+1}/{len(files)}] {filename}...", end=" ")
            status, msg = process_single_image(os.path.join(INPUT_FOLDER, filename), OUTPUT_FOLDER)
            log.write(f"{filename} | {'OK' if status else 'FAIL'} | {msg}\n")
            print("-> OK" if status else f"-> FAIL ({msg})")

if __name__ == "__main__":
    main()