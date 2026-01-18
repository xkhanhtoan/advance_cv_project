import os
import json
from tqdm import tqdm
from typing import Dict, List
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image

# === IMPORT SANITIZER ===
try:
    from sanitizer import SmartSanitizer
except ImportError:
    class SmartSanitizer:
        @staticmethod
        def sanitize(x): return x
# ========================

class SROIEInference:
    def __init__(self, model_path: str, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        
        if hasattr(self.model.config, 'label2id'):
            self.id2label = self.model.config.id2label
        else:
            self.label2id = {
                "O": 0, "B-COMPANY": 1, "I-COMPANY": 2, "B-DATE": 3, "I-DATE": 4,
                "B-ADDRESS": 5, "I-ADDRESS": 6, "B-TOTAL": 7, "I-TOTAL": 8
            }
            self.id2label = {v: k for k, v in self.label2id.items()}
    
    def predict_single(self, image_path: str, words: List[str], boxes: List[List[int]]) -> Dict[str, str]:
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        normalized_boxes = [[int(1000 * b[0] / w), int(1000 * b[1] / h),
                             int(1000 * b[2] / w), int(1000 * b[3] / h)] for b in boxes]
        
        encoding = self.processor(image, words, boxes=normalized_boxes, truncation=True,
                                  padding='max_length', max_length=512, return_tensors='pt')
        
        input_data = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**input_data)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        word_ids = encoding.word_ids(batch_index=0)
        word_predictions = []
        prev_idx = None
        for i, w_idx in enumerate(word_ids):
            if w_idx is not None and w_idx != prev_idx:
                word_predictions.append({'word': words[w_idx], 'label': self.id2label[predictions[i]]})
                prev_idx = w_idx
        
        return self.extract_entities(word_predictions)
    
    def extract_entities(self, word_predictions: List[Dict]) -> Dict[str, str]:
        entities = {'company': '', 'date': '', 'address': '', 'total': ''}
        current_entity, current_text = None, []
        for item in word_predictions:
            word, label = item['word'], item['label']
            if label.startswith('B-'):
                if current_entity: entities[current_entity.lower()] = ' '.join(current_text)
                current_entity, current_text = label[2:], [word]
            elif label.startswith('I-') and label[2:] == current_entity:
                current_text.append(word)
            else:
                if current_entity: entities[current_entity.lower()] = ' '.join(current_text)
                current_entity, current_text = None, []
        if current_entity: entities[current_entity.lower()] = ' '.join(current_text)
        return entities

    def predict_from_box_file(self, image_path: str, box_file_path: str) -> Dict[str, str]:
        words, boxes = [], []
        with open(box_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                p = line.strip().split(',')
                if len(p) >= 9:
                    try:
                        boxes.append([int(p[0]), int(p[1]), int(p[4]), int(p[5])])
                        words.append(','.join(p[8:]))
                    except: continue
        return self.predict_single(image_path, words, boxes)

class SROIEEvaluator:
    def __init__(self, model_path: str, test_dir: str):
        self.inferencer = SROIEInference(model_path)
        self.test_dir = test_dir
    
    def normalize(self, text: str) -> str:
        return text.strip().upper().replace(' ', '')
    
    def evaluate_all(self) -> Dict:
        box_dir = os.path.join(self.test_dir, 'box')
        sample_ids = [f.replace('.txt', '') for f in os.listdir(box_dir) if f.endswith('.txt')]
        
        metrics = {f: {'correct': 0, 'partial': 0.0} for f in ['company', 'date', 'address', 'total']}
        detailed = {}
        
        print(f"Evaluating {len(sample_ids)} samples...")
        for sid in tqdm(sample_ids):
            # 1. Raw Prediction
            raw_pred = self.inferencer.predict_from_box_file(
                os.path.join(self.test_dir, 'img', f'{sid}.jpg'),
                os.path.join(self.test_dir, 'box', f'{sid}.txt')
            )
            
            # 2. Sanitize
            clean_pred = SmartSanitizer.sanitize(raw_pred)
            
            # 3. Load GT
            with open(os.path.join(self.test_dir, 'entities', f'{sid}.txt'), 'r', encoding='utf-8', errors='ignore') as f:
                gt = json.load(f)
            
            res = {}
            for k in metrics:
                r_val = raw_pred.get(k, '') # Giá trị gốc
                c_val = clean_pred.get(k, '') # Giá trị đã sạch
                g_val = gt.get(k, '') # Giá trị đúng
                
                c_norm, g_norm = self.normalize(c_val), self.normalize(g_val)
                is_exact = c_norm == g_norm
                
                if is_exact: metrics[k]['correct'] += 1
                
                p_set, g_set = set(c_norm), set(g_norm)
                u = len(p_set | g_set)
                score = len(p_set & g_set) / u if u > 0 else 0.0
                metrics[k]['partial'] += score
                
                # Lưu cả Raw, Clean và GT để so sánh
                res[k] = {
                    'exact': is_exact, 'score': score, 
                    'raw': r_val, 'clean': c_val, 'gt': g_val
                }
            detailed[sid] = res
            
        # Summary
        total = len(sample_ids)
        summary = {'total': total, 'fields': {}}
        print("\n" + "="*50 + "\nRESULTS\n" + "="*50)
        for k, v in metrics.items():
            acc = v['correct'] / total * 100
            p_score = v['partial'] / total * 100
            summary['fields'][k] = {'acc': acc, 'partial': p_score}
            print(f"{k.upper():<10} Exact: {acc:.2f}% | Partial: {p_score:.2f}%")
        
        print("-" * 50 + f"\nAVG EXACT: {(sum(m['acc'] for m in summary['fields'].values())/4):.2f}%")
        return {'summary': summary, 'details': detailed}

    def save_results(self, results, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    MODEL_PATH = '../train/layoutlmv3_sroie_output/best_model'
    TEST_DIR = './sroie-dataset/SROIE2019/test'
    OUTPUT_FILE = './outputs/evaluation_results.json'
    
    evaluator = SROIEEvaluator(MODEL_PATH, TEST_DIR)
    results = evaluator.evaluate_all()
    
    print("\n" + "="*50 + "\nTOP ERRORS (DEBUG VIEW)\n" + "="*50)
    # Sắp xếp theo lỗi tệ nhất
    sorted_err = sorted(results['details'].items(), key=lambda x: sum(y['score'] for y in x[1].values()))
    
    for sid, res in sorted_err[:10]: # Xem 8 lỗi đầu tiên
        print(f"\n[Sample: {sid}]")
        for k, v in res.items():
            if not v['exact']:
                # In ra định dạng dễ nhìn để so sánh Raw -> Clean
                print(f"  {k.upper()}:")
                print(f"    Raw  : '{v['raw']}'")
                print(f"    Clean: '{v['clean']}'")
                print(f"    GT   : '{v['gt']}'")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    evaluator.save_results(results, OUTPUT_FILE)

if __name__ == "__main__":
    main()