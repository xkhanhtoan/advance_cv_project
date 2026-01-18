import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
from typing import Dict, List
import json
import os

# === IMPORT SANITIZER (BẮT BUỘC) ===
try:
    from utils.sanitizer import SmartSanitizer
except ImportError:
    print("CẢNH BÁO: Không tìm thấy file sanitizer.py. Kết quả sẽ không được làm sạch!")
    # Fallback đummy để code không crash nếu thiếu file
    class SmartSanitizer:
        @staticmethod
        def sanitize(x): return x
# ===================================

class SROIEInference:
    """Inference cho LayoutLMv3 trên SROIE receipts (kèm Auto-Cleaning)"""
    
    def __init__(self, model_path: str, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model và processor
        print(f"Loading model from {model_path} to {self.device}...")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_path,
            apply_ocr=False
        )
        
        # Load label mappings
        label_file = os.path.join(model_path, 'label2id.json')
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.label2id = json.load(f)
            self.id2label = {int(v): k for k, v in self.label2id.items()}
        else:
            # Fallback nếu không thấy file config (dùng mapping mặc định SROIE)
            self.label2id = {
                "O": 0, "B-COMPANY": 1, "I-COMPANY": 2, "B-DATE": 3, "I-DATE": 4,
                "B-ADDRESS": 5, "I-ADDRESS": 6, "B-TOTAL": 7, "I-TOTAL": 8
            }
            self.id2label = {v: k for k, v in self.label2id.items()}
    
    def predict_single(self, image_path: str, words: List[str], boxes: List[List[int]]) -> Dict[str, str]:
        # Load image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # Normalize boxes (0-1000)
        normalized_boxes = [
            [
                int(1000 * box[0] / width),
                int(1000 * box[1] / height),
                int(1000 * box[2] / width),
                int(1000 * box[3] / height)
            ]
            for box in boxes
        ]
        
        # Encode
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Lấy word_ids để map lại kết quả token -> word gốc
        word_ids = encoding.word_ids(batch_index=0)
        
        # Move to device
        input_data = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**input_data)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        predictions = predictions.cpu().numpy()[0]
        
        # Align predictions với original words
        word_predictions = []
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                word_predictions.append({
                    'word': words[word_idx],
                    'label': self.id2label[predictions[idx]]
                })
                previous_word_idx = word_idx
        
        # Trích xuất và TỰ ĐỘNG LÀM SẠCH
        entities = self.extract_entities(word_predictions)
        return entities
    
    def extract_entities(self, word_predictions: List[Dict]) -> Dict[str, str]:
        """Convert B-I-O labels to text entities and apply Sanitizer"""
        entities = {
            'company': '',
            'date': '',
            'address': '',
            'total': ''
        }
        
        current_entity = None
        current_text = []
        
        for item in word_predictions:
            word = item['word']
            label = item['label']
            
            if label.startswith('B-'):
                if current_entity and current_text:
                    entity_name = current_entity.lower()
                    entities[entity_name] = ' '.join(current_text)
                
                current_entity = label[2:]
                current_text = [word]
                
            elif label.startswith('I-'):
                entity_name = label[2:]
                if entity_name == current_entity:
                    current_text.append(word)
                else:
                    # Trường hợp model nhảy I- mà không có B-, ta vẫn gom vào nếu đúng loại
                    # Hoặc reset bắt đầu mới (tùy chiến thuật, ở đây chọn reset cho an toàn)
                    if current_entity and current_text:
                         entities[current_entity.lower()] = ' '.join(current_text)
                    current_entity = entity_name
                    current_text = [word]
                    
            else: # Label 'O'
                if current_entity and current_text:
                    entity_name = current_entity.lower()
                    entities[entity_name] = ' '.join(current_text)
                current_entity = None
                current_text = []
        
        # Xử lý cụm cuối cùng còn sót lại
        if current_entity and current_text:
            entity_name = current_entity.lower()
            entities[entity_name] = ' '.join(current_text)
        
        # === BƯỚC QUAN TRỌNG NHẤT: SANITIZE ===
        # Gọi class SmartSanitizer để làm sạch dữ liệu
        clean_entities = SmartSanitizer.sanitize(entities)
        
        return clean_entities
    
    def predict_from_box_file(self, image_path: str, box_file_path: str) -> Dict[str, str]:
        words = []
        boxes = []
        
        # Thêm errors='ignore' để tránh crash với các file text lạ
        with open(box_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split(',')
                # Format SROIE: x1,y1,x2,y2,x3,y3,x4,y4,text
                if len(parts) >= 9:
                    # Lấy toạ độ Top-Left (0,1) và Bottom-Right (4,5)
                    try:
                        box = [int(parts[0]), int(parts[1]), 
                               int(parts[4]), int(parts[5])]
                        # Nối lại phần text (phòng trường hợp text có dấu phẩy)
                        text = ','.join(parts[8:])
                        
                        words.append(text)
                        boxes.append(box)
                    except ValueError:
                        continue
        
        return self.predict_single(image_path, words, boxes)

if __name__ == '__main__':
    # Init model
    infer = SROIEInference(model_path=r'..\train\layoutlmv3_sroie_output/best_model')

    result = infer.predict_from_box_file(
        image_path=r'../SROIE2019/test/img/X00016469670.jpg',
        box_file_path=r'../SROIE2019/test/box/X00016469670.txt'
    )

    print(result)
    # {'company': 'STARBUCKS COFFEE', 'date': '20/10/2018', ...}