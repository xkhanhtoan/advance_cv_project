import re
from typing import Dict

class SmartSanitizer:
    @staticmethod
    def sanitize(entities: Dict[str, str]) -> Dict[str, str]:
        """
        ULTIMATE MIX VERSION:
        - DATE: Logic V1 (List Patterns) -> Uy tín nhất (98%).
        - TOTAL: Logic V5 (Last Token + Rstrip) -> Uy tín nhất (88%).
        - COMPANY: Logic V5 (Suffix Check) -> Uy tín nhất (83%).
        """
        clean = entities.copy()

        # ==========================================
        # 1. TOTAL (Giữ logic V5 đang tốt nhất)
        # ==========================================
        if 'total' in clean and clean['total']:
            val = clean['total'].strip()
            parts = val.split()
            found_total = None
            
            # Quét từ cuối lên
            for part in reversed(parts):
                if any(char.isdigit() for char in part):
                    found_total = part
                    break
            
            if found_total:
                # Xóa dấu câu thừa ở đuôi
                val_clean = found_total.strip(".,-")
                # Xóa chữ cái rác dính ở đuôi (ví dụ: 5.80SR -> 5.80)
                val_clean = re.sub(r'[^\d]+$', '', val_clean)
                clean['total'] = val_clean

        # ==========================================
        # 2. DATE (Khôi phục logic V1/V2 cũ - Đạt 98%)
        # ==========================================
        if 'date' in clean and clean['date']:
            val = clean['date']
            
            # Danh sách pattern bắt dính tốt nhất từng dùng
            patterns = [
                # 22 MAR 2018 (Ngày tháng chữ)
                r'\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}\b', 
                # 15/12/2017 (Ngày tháng số phổ biến)
                r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b', 
                # 2018-05-09 (ISO format)
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'      
            ]
            
            found = False
            for pat in patterns:
                # Dùng Regex Search thông thường (mạnh hơn việc chia if/else cứng nhắc)
                match = re.search(pat, val, re.IGNORECASE)
                if match:
                    result = match.group(0).upper()
                    # Fix lỗi khoảng trắng trong ngày tháng số: "20 / 10 / 2018" -> "20/10/2018"
                    # Nếu pattern không chứa chữ cái (tức là chỉ có số và dấu), thì xóa space
                    if not re.search(r'[A-Z]', result):
                        result = result.replace(' ', '')
                    else:
                         # Nếu là ngày có chữ (22 MAR 2018), chỉ chuẩn hóa space
                        result = re.sub(r'\s+', ' ', result)
                    
                    clean['date'] = result
                    found = True
                    break
            
            if not found:
                # Fallback clean rác nếu không khớp pattern nào
                val = val.upper()
                val = re.sub(r'(DATE|TIME|RECEIPT|INVOICE|DATED|BILL|DT|SERVED BY)[:\s]*', '', val)
                val = re.sub(r'\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?', '', val)
                val = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', val) 
                # Nếu còn lại là chuỗi số có khoảng trắng (vd: 20 10 2018), ép xóa space lần cuối
                if re.match(r'^[\d/\.\-\s]+$', val):
                     val = val.replace(' ', '')
                clean['date'] = val.strip()

        # ==========================================
        # 3. COMPANY (Giữ logic V5 đang tốt nhất)
        # ==========================================
        if 'company' in clean and clean['company']:
            val = clean['company']
            # Bỏ mã số thuế trong ngoặc
            val = re.sub(r'\s*\(\s*[\w\d\-\.]+\s*\)$', '', val)
            
            # Giữ dấu chấm cho các từ viết tắt công ty
            suffixes = ('BHD.', 'SDN.', 'LTD.', 'PLT.', 'INC.', 'CO.')
            if val.strip().upper().endswith(suffixes):
                 val = val.rstrip(',') # Chỉ xóa dấu phẩy thừa
            else:
                val = re.sub(r'[,\.]+$', '', val) # Xóa sạch dấu câu cuối
                
            clean['company'] = val.strip()
            
        return clean