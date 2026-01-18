import torch
from torch.optim import AdamW
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import os
import json
import random

# Import module của bạn
from data_preparation import SROIEDataProcessor
from dataset import create_dataloaders

# --- Utility: Set Seed để kết quả nhất quán ---
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

class LayoutLMv3Trainer:
    def __init__(self, model, processor, train_loader, val_loader, 
                 device, label2id, id2label, output_dir='./outputs'):
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def train_epoch(self, optimizer, scheduler, epoch_idx):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {epoch_idx+1}')
        
        for batch in progress_bar:
            # Move data to GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (tránh lỗi gradient quá lớn)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                # Decode labels (loại bỏ padding -100)
                for pred_seq, label_seq in zip(predictions, labels):
                    valid_indices = label_seq != -100
                    
                    # Map ID về String (B-COMPANY, ...)
                    pred_labels = [self.id2label[p] for p in pred_seq[valid_indices]]
                    true_labels = [self.id2label[l] for l in label_seq[valid_indices]]
                    
                    all_predictions.append(pred_labels)
                    all_labels.append(true_labels)
        
        # Calculate metrics
        return {
            'loss': total_loss / len(self.val_loader),
            'f1': f1_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions),
            'report': classification_report(all_labels, all_predictions)
        }
    
    def train(self, num_epochs=20, learning_rate=2e-5, warmup_ratio=0.1):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        best_f1 = 0
        patience = 0
        patience_limit = 5 # Dừng sớm nếu sau 5 epoch không khá hơn
        
        print(f"Start Training on {self.device}")
        
        for epoch in range(num_epochs):
            # 1. Train
            train_loss = self.train_epoch(optimizer, scheduler, epoch)
            
            # 2. Evaluate
            metrics = self.evaluate()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f}")
            print(f"F1: {metrics['f1']:.4f}")
            
            # 3. Save Best Model
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                patience = 0
                print(f"🚀 New Best F1! Saving model...")
                self.save_model("best_model")
                print(metrics['report']) # In chi tiết khi có kỷ lục mới
            else:
                patience += 1
                print(f"No improvement. Patience: {patience}/{patience_limit}")
            
            if patience >= patience_limit:
                print("Early stopping triggered!")
                break
                
        print(f"\nTraining Finished. Best F1: {best_f1}")
    
    def save_model(self, name):
        path = os.path.join(self.output_dir, name)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        # Lưu label map để dùng khi inference
        with open(os.path.join(path, "label_map.json"), "w") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, f)

def main():
    set_seed(42)
    
    # Cấu hình
    DATA_DIR = "./sroie-dataset/SROIE2019" 
    OUTPUT_DIR = "layoutlmv3_sroie_output"
    BATCH_SIZE = 4       
    NUM_EPOCHS = 20
    LEARNING_RATE = 2e-5 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare Data
    print("Loading Data...")
    data_proc = SROIEDataProcessor(DATA_DIR)
    full_data = data_proc.load_dataset('train')
    
    # Split Train/Val (90/10 vì data ít và sạch)
    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # 2. Model & Processor
    model_id = "microsoft/layoutlmv3-base"
    processor = LayoutLMv3Processor.from_pretrained(model_id, apply_ocr=False)
    
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(data_proc.label2id),
        id2label=data_proc.id2label,
        label2id=data_proc.label2id
    )
    
    # 3. Create Dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, processor, batch_size=BATCH_SIZE
    )
    
    # 4. Train
    trainer = LayoutLMv3Trainer(
        model, processor, train_loader, val_loader, 
        device, data_proc.label2id, data_proc.id2label, 
        output_dir=OUTPUT_DIR
    )
    
    trainer.train(num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

if __name__ == "__main__":
    main()