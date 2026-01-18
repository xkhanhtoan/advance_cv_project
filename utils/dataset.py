import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class SROIEDataset(Dataset):
    def __init__(self, data, processor, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- FIX LỖI Ở ĐÂY: Logic linh hoạt để lấy ảnh ---
        # 1. Tìm nguồn ảnh từ các key có thể có
        if 'image' in item:
            img_source = item['image']
        elif 'image_path' in item:
            img_source = item['image_path']
        elif 'img_path' in item:
            img_source = item['img_path']
        else:
            raise KeyError(f"Lỗi ở mẫu số {idx}: Không tìm thấy thông tin ảnh. Các key hiện có: {list(item.keys())}")

        # 2. Xử lý ảnh (Path string hoặc PIL Object)
        try:
            if isinstance(img_source, str):
                # Nếu là đường dẫn (string) -> Open
                image = Image.open(img_source).convert("RGB")
            else:
                # Nếu đã là PIL Image object -> Chỉ cần convert màu
                image = img_source.convert("RGB")
        except Exception as e:
            print(f"Lỗi khi mở ảnh tại index {idx}. Source: {img_source}")
            raise e

        words = item['words']
        boxes = item['boxes']
        word_labels = item['labels'] 

        # 3. Xử lý qua Processor
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 4. Loại bỏ chiều batch
        return {k: v.squeeze() for k, v in encoding.items()}

def create_dataloaders(train_data, val_data, processor, batch_size=4, max_length=512):
    """
    Tạo DataLoader cho train và validation
    """
    train_dataset = SROIEDataset(train_data, processor, max_length)
    val_dataset = SROIEDataset(val_data, processor, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader