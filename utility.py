import cv2
import numpy as np
import re
from collections import defaultdict, deque, Counter
import difflib
import os
import tempfile
from fast_plate_ocr import LicensePlateRecognizer

# Initialize fast-plate-ocr model (CPU by default)
OCR_MODEL = LicensePlateRecognizer('cct-xs-v1-global-model')

plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

# OCR Text Correction Function
def correct_plate_format(ocr_text):
    """
    Correct common OCR misreads in license plate text.
    """
    mapping_num_to_alpha = {"0":"O", "1":"I", "2":"Z", "5":"S", "6":"G", "8":"B"}
    mapping_alpha_to_num = {"O":"0", "I":"1", "Z":"2", "S":"5", "G":"6", "B":"8"}

    ocr_text = ocr_text.upper()
    ocr_text = ocr_text.replace(" ", "").replace("-", "").replace("_", "")

    # print(f"After cleaning in correct_plate_format: '{ocr_text}', length: {len(ocr_text)}")
    
    if len(ocr_text) != 7:  # Format: LLNNLLL
        return ""
    
    corrected = []
    for i, char in enumerate(ocr_text):
        if i < 2 or (i >= 4 and i <= 6):  # Letter positions
            if char.isdigit() and char in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[char])
            elif char.isalpha():
                corrected.append(char)
            else:
                # print(f"Invalid char in letter pos {i}: '{char}'")
                return ""  # Invalid character
        elif i == 2 or i == 3:  # Number positions
            if char.isalpha() and char in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[char])
            elif char.isdigit():
                corrected.append(char)
            else:
                # print(f"Invalid char in num pos {i}: '{char}'")
                return ""  # Invalid character
            
    corrected_text = "".join(corrected)
    # print(f"Corrected text: '{corrected_text}', pattern match: {bool(plate_pattern.match(corrected_text))}")
    return corrected_text

# Recognize Text on Cropped Plate
def recognize_plate(plate_crop):
    '''
    Recognize and correct license plate text from cropped image using fast-plate-ocr.
    '''
    if plate_crop is None or plate_crop.size == 0:
        # print("Empty plate crop provided")
        return ""
    
    # print(f"Plate crop shape: {plate_crop.shape}, size: {plate_crop.size}")
    
    temp_path = None
    try:
        # Save crop to temporary file (run() expects a path)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, plate_crop)
        
        # Perform OCR using the CPU-accelerated fast-plate-ocr model
        ocr_result = OCR_MODEL.run(temp_path)
        print(f"Raw OCR result: {ocr_result}")
        
        # Clean up temp file immediately
        os.unlink(temp_path)
        temp_path = None
        
        # Handle if result is a list (e.g., [text]); extract string
        if isinstance(ocr_result, list):
            ocr_text = ocr_result[0] if ocr_result else ""
        else:
            ocr_text = ocr_result or ""
        
        print(f"Extracted ocr_text: '{ocr_text}'")
        
        if ocr_text:
            corrected_text = correct_plate_format(ocr_text)
            if plate_pattern.match(corrected_text):
                # print(f"Final recognized plate: '{corrected_text}'")
                return corrected_text
                
    except Exception as e:
        print(f"OCR error: {e}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # print("No valid plate recognized")
    return ""

# Stabilizes the detected plate text using historical data and voting mechanism.
def get_stable_plate(box_id, plate_text, history, final, pattern):
    # print(f"Stabilizing for box_id '{box_id}': input plate_text='{plate_text}'")
    
    if plate_text:
        history[box_id].append(plate_text)
    
    # Get the list of recent plate texts
    recent_plates = list(history[box_id])
    # print(f"Recent plates for {box_id}: {recent_plates}")
    
    if not recent_plates:
        return ""
    
    # Group similar plates (similarity > 70%)
    groups = []
    for plate in recent_plates:
        found = False
        for group in groups:
            rep = group[0]  # Use first plate as representative
            similarity = difflib.SequenceMatcher(None, plate, rep).ratio()
            if similarity > 0.7:
                group.append(plate)
                found = True
                break
        if not found:
            groups.append([plate])
    
    # print(f"Similarity groups: {groups}")
    
    # For each group, perform per-character voting
    group_stables = []
    for group in groups:
        if not group:
            continue
        max_len = max(len(p) for p in group) # max_len = 7 from corrected_plate_format function
        padded_plates = [p.ljust(max_len) for p in group]
        stable_chars = []
        for pos in range(max_len):
            chars_at_pos = [plate[pos] for plate in padded_plates]
            most_common_char = Counter(chars_at_pos).most_common(1)
            stable_chars.append(most_common_char[0][0] if most_common_char else ' ')
        stable_plate = ''.join(stable_chars).rstrip()
        group_stables.append(stable_plate)
    
    # Choose the most frequent resulting plate among all groups
    if group_stables:
        most_frequent_stable = Counter(group_stables).most_common(1)[0][0]
        final[box_id] = most_frequent_stable  # Update final dict
        # print(f"Stabilized plate for {box_id}: '{most_frequent_stable}'")
        return most_frequent_stable
    
    # print(f"No stable plate for {box_id}")
    return ""

if __name__ == "__main__":
    pass