# invoice-ocr-pipeline

Dự án nhận dạng hoá đơn, trích xuất văn bản và tạo dữ liệu có cấu trúc từ hình ảnh hoá đơn.

## Giới thiệu

Mục tiêu của dự án là xây dựng một pipeline xử lý hoá đơn hoàn chỉnh, bao gồm thu thập dữ liệu, tiền xử lý ảnh, phát hiện vùng văn bản, OCR và trích xuất thông tin quan trọng như tên cửa hàng, ngày tháng, tổng tiền, địa chỉ...

---

## Pipeline chi tiết

### **1. Thu thập & chuẩn hoá dữ liệu**

#### Bộ dữ liệu sử dụng

##### **SROIE Dataset V2 – Kaggle**

Nguồn: [SROIE Dataset V2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2)

Dùng để huấn luyện mô hình YOLO nhận diện các vùng text cơ bản.

##### **Bộ dữ liệu tự thu thập**

Nguồn: [RECEIPT_OCR](https://drive.google.com/drive/folders/1o9z7aHz03oNXBfJ1zRomDiLqBSRB2ax2?usp=sharing)

Bao gồm nhiều loại hoá đơn chụp thực tế, giúp mô hình tổng quát tốt hơn.

#### Công cụ gán nhãn

Sử dụng **Label Studio** để gán nhãn và trực quan hoá dữ liệu.


### **2. Tiền xử lý ảnh (Preprocessing)**

* **Định vị (Localization):** Dùng *Thresholding* + *MinAreaRect* để tách hóa đơn khỏi nền.
* **Cắt phẳng (Warping):** Dùng *Perspective Transform* để đưa ảnh về góc nhìn chuẩn.
* **Xoay thẳng (Deskewing):** Áp dụng *Hough Lines* để căn chỉnh văn bản về ngang.
* **Làm sạch (Binarization):** Dùng *Adaptive Threshold* để tạo ảnh nhị phân sạch cho OCR.


### **3. Object Detection (YOLO)**

* Dùng YOLO phát hiện các vùng chứa văn bản.
* Fine-tune mô hình bằng hai tập dữ liệu trên.


### **4. OCR (in progress)**

Đang thực hiện.


### **5. Information Extraction (in progress)**

Đang thực hiện.

---

## Cài đặt môi trường

Cài đặt thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

## Fine-tune YOLO với dataset SROIE

* Để chuyển dữ liệu SROIE sang định dạng JSONL tương thích với Label Studio, sử dụng script:

```bash
python ./scripts/convert_sroie_to_label_studio.py
```

Kết quả lưu tại [sroie_train.json](sroie_train.json) và [sroie_test.json](sroie_test.json)

* Import dataset vào label-studio theo hướng dẫn tại [label-studio-guide.pdf](documents/label-studio-guide.pdf)
* Chuẩn bị dữ liệu cho YOLO
```bash
python ./scripts/train_val_split.py
```
```bash
python ./scripts/fix_class_id.py
```
* Train model
```bash
python ./train/sroie_yolov8_finetune.py
```
```bash
python ./train/sroie_yolov9c_finetune.py
```

## Gán nhãn cho dữ liệu ảnh chụp hoá đơn và fine-tune trên dữ liệu này
* Tiền xử lý ảnh
```bash
python ./data_preprocessing/preprocessing.py
```
* Xử dụng mô hình YOLO đã fine-tune để gán nhãn tự động
```bash
python ./data_preprocessing/data_labelling.py
```

* Import dữ liệu đã được gán nhãn vào label-studio để check lại theo guideline ở [label-studio-guide.pdf](documents/label-studio-guide.pdf)

* Fine-tune dữ liệu theo pineline tương tự với SROIE dataset, kiểm tra đường dẫn chính xác trước khi chạy chương trình
* Chuẩn bị dữ liệu cho YOLO
```bash
python ./scripts/train_val_split.py
```
```bash
python ./scripts/fix_class_id.py
```
* Train model
```bash
python ./train/receipt_yolov8_finetune.py
```
```bash
python ./train/receipt_yolov9c_finetune.py
```

* Đánh giá chương trình: [eval.ipynb](evaluate/eval.ipynb)

## Phân công công việc
* Hoàng Lê Tuấn: Thu thập dữ liệu
* Nguyễn Khánh Toán: Tiền xử lý dữ liệu
* Phạm Văn Vinh: Finetune mô hình
