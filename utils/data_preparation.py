import os
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
from PIL import Image
from fuzzysearch import find_near_matches

class SROIEDataProcessor:
    """Xử lý dữ liệu SROIE cho LayoutLMv3 - Version 4 - Better matching"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.label2id = {
            'O': 0,
            'B-COMPANY': 1,
            'I-COMPANY': 2,
            'B-DATE': 3,
            'I-DATE': 4,
            'B-ADDRESS': 5,
            'I-ADDRESS': 6,
            'B-TOTAL': 7,
            'I-TOTAL': 8
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def read_box_file(self, box_path: str) -> List[Tuple[List[int], str]]:
        boxes = []
        with open(box_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    bbox = [int(parts[0]), int(parts[1]), int(parts[4]), int(parts[5])]
                    text = ','.join(parts[8:])
                    boxes.append((bbox, text))
        return boxes
    
    def read_entities_file(self, entities_path: str) -> Dict[str, str]:
        with open(entities_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_box(self, box: List[int], width: int, height: int) -> List[int]:
        return [
            int(1000 * box[0] / width),
            int(1000 * box[1] / height),
            int(1000 * box[2] / width),
            int(1000 * box[3] / height)
        ]
    
    def is_word_boundary_match(self, word: str, entity: str) -> bool:
        """
        Check if word appears as a complete token in entity (with word boundaries)
        This prevents '2' from matching '32' in 'NO.32'
        """
        word_upper = word.upper()
        entity_upper = entity.upper()
        
        # Try to find word with word boundaries
        import re
        # Use word boundary regex: \b for alphanumeric, or space/punctuation
        pattern = r'(^|[\s,\.;:])' + re.escape(word_upper) + r'($|[\s,\.;:])'
        
        return re.search(pattern, entity_upper) is not None
    
    def assign_labels(self, boxes: List[Tuple[List[int], str]], entities: Dict[str, str]) -> List[int]:
            """
            Advanced Label Assignment: Nối chuỗi -> Fuzzy Search -> Map ngược lại Word Index
            """
            words = [b[1] for b in boxes]
            # 1. Tạo chuỗi văn bản đầy đủ từ các từ OCR
            # Lưu ý: Cần theo dõi vị trí ký tự để map ngược lại index của từ
            full_text = ""
            # word_indices[i] sẽ trả về index của từ tương ứng với ký tự thứ i trong full_text
            word_indices = [] 
            
            for idx, word in enumerate(words):
                full_text += word + " " # Thêm dấu cách
                word_indices.extend([idx] * (len(word) + 1)) # +1 cho dấu cách
                
            labels = [0] * len(words) # 0 là 'O'
            
            # Thứ tự ưu tiên: Match cái dài & khó trước (Address/Company) -> Ngắn sau (Date/Total)
            # Để tránh việc từ ngắn ghi đè lên từ dài
            target_entities = ['address', 'company', 'date', 'total']
            
            for entity_type in target_entities:
                entity_text = entities.get(entity_type, "")
                if not entity_text: continue
                
                # Cấu hình độ sai số (Dist) cho phép
                # Address/Company dài -> cho phép sai nhiều (20% độ dài)
                # Date/Total ngắn -> cho phép sai ít (tối đa 1-2 ký tự)
                max_dist = int(len(entity_text) * 0.2)
                if entity_type in ['date', 'total']:
                    max_dist = min(max_dist, 2) 
                
                # Tìm chuỗi gần đúng nhất trong full_text
                # find_near_matches trả về list các matches
                matches = find_near_matches(
                    entity_text.upper(), 
                    full_text.upper(), 
                    max_l_dist=max_dist
                )
                
                if matches:
                    # Lấy match tốt nhất (có dist nhỏ nhất)
                    best_match = sorted(matches, key=lambda x: x.dist)[0]
                    
                    # Map từ ký tự start/end về danh sách các từ (words)
                    start_char, end_char = best_match.start, best_match.end
                    
                    # Lấy tập hợp các index của từ nằm trong vùng match
                    matched_word_idxs = set()
                    for char_i in range(start_char, end_char):
                        if char_i < len(word_indices):
                            matched_word_idxs.add(word_indices[char_i])
                    
                    # Sắp xếp index để gán BIO
                    sorted_idxs = sorted(list(matched_word_idxs))
                    if not sorted_idxs: continue
                    
                    # Gán nhãn B- (Begin)
                    b_idx = sorted_idxs[0]
                    # Chỉ gán nếu chưa có nhãn (hoặc bạn có thể quy định độ ưu tiên ghi đè)
                    if labels[b_idx] == 0:
                        labels[b_idx] = self.label2id[f'B-{entity_type.upper()}']
                    
                    # Gán nhãn I- (Inside) cho các từ còn lại
                    for i_idx in sorted_idxs[1:]:
                        if labels[i_idx] == 0:
                            labels[i_idx] = self.label2id[f'I-{entity_type.upper()}']
    
            return labels    

    def process_single_sample(self, sample_id: str, split: str = 'train') -> Dict:
        base_path = os.path.join(self.data_dir, split)
        
        img_path = os.path.join(base_path, 'img', f'{sample_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        
        box_path = os.path.join(base_path, 'box', f'{sample_id}.txt')
        boxes_with_text = self.read_box_file(box_path)
        
        labels = None
        if split == 'train':
            entities_path = os.path.join(base_path, 'entities', f'{sample_id}.txt')
            entities = self.read_entities_file(entities_path)
            labels = self.assign_labels(boxes_with_text, entities)
        
        boxes = [bbox for bbox, _ in boxes_with_text]
        words = [text for _, text in boxes_with_text]
        normalized_boxes = [self.normalize_box(box, width, height) for box in boxes]
        
        return {
            'image': image,
            'words': words,
            'boxes': normalized_boxes,
            'labels': labels,
            'id': sample_id
        }
    
    def load_dataset(self, split: str = 'train') -> List[Dict]:
        base_path = os.path.join(self.data_dir, split, 'box')
        sample_ids = [f.replace('.txt', '') for f in os.listdir(base_path) 
                     if f.endswith('.txt')]
        
        print(f"Loading {len(sample_ids)} samples from {split} set...")
        
        dataset = []
        for sample_id in sample_ids:
            try:
                sample = self.process_single_sample(sample_id, split)
                dataset.append(sample)
            except Exception as e:
                print(f"Error processing {sample_id}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(dataset)} samples")
        return dataset


if __name__ == "__main__":
    processor = SROIEDataProcessor("./sroie-dataset/SROIE2019")
    train_data = processor.load_dataset('train')
    
    sample = train_data[0]
    print(f"\nSample ID: {sample['id']}")
    
    from collections import Counter
    label_counts = Counter(sample['labels'])
    print("\nLabel distribution:")
    for label_id in sorted(label_counts.keys()):
        label_name = processor.id2label[label_id]
        count = label_counts[label_id]
        print(f"  {label_name:15s}: {count:3d} ({count/len(sample['labels'])*100:.1f}%)")
    
    print("\nLabeled words:")
    for i, (word, label_id) in enumerate(zip(sample['words'], sample['labels'])):
        label_name = processor.id2label[label_id]
        if label_name != 'O':
            print(f"{i:2d}. {word:50s} -> {label_name}")
