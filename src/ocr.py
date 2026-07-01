"""
OCR Module for detecting elevation values in maps
"""
import numpy as np
import cv2
import pytesseract
import re
import os

# Ρύθμιση path για Tesseract (Windows)
if os.name == 'nt':  # Windows
    try:
        # Προσπάθησε να βρει το Tesseract στις συνήθεις θέσεις
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    except:
        pass

class OCRDetector:
    """Detect elevation values using OCR"""
    
    def __init__(self):
        self.confidence_threshold = 60
        self.psm = 6  # Assume a single uniform text block
        
    def detect_elevations(self, image):
        """Detect elevation numbers in image"""
        if image is None:
            return []
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Preprocess for OCR
        processed = self._preprocess_image(gray)
        
        try:
            # Get text data
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"OCR error: {e}")
            print("Make sure Tesseract is installed:")
            print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            return []
        
        elevations = []
        for i, text in enumerate(data['text']):
            if not text.strip():
                continue
                
            # Try to parse as number
            try:
                # Remove non-numeric characters
                clean_text = re.sub(r'[^0-9.]', '', text)
                if clean_text:
                    value = float(clean_text)
                    
                    # Check if it's a reasonable elevation value
                    if 0 <= value <= 5000:  # Range for Greece
                        conf = int(data['conf'][i])
                        if conf >= self.confidence_threshold:
                            x = data['left'][i]
                            y = data['top'][i]
                            w = data['width'][i]
                            h = data['height'][i]
                            
                            elevations.append({
                                'value': value,
                                'bbox': (x, y, w, h),
                                'confidence': conf
                            })
            except:
                continue
        
        return elevations
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Apply thresholding
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if text is light on dark background
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def map_elevations_to_contours(self, contours, elevations, max_distance=50):
        """Assign elevations to closest contours"""
        if not elevations or not contours:
            return [0] * len(contours)
        
        # Calculate centroid of each contour
        centroids = []
        for contour in contours:
            if len(contour) == 0:
                centroids.append((0, 0))
                continue
            center = np.mean(contour, axis=0)
            centroids.append(center)
        
        # Assign each elevation to closest contour
        assignments = [0] * len(contours)
        distances = [float('inf')] * len(contours)
        
        for elev in elevations:
            # Get centroid of text bounding box
            x, y, w, h = elev['bbox']
            text_center = (y + h/2, x + w/2)
            
            # Find closest contour
            for i, centroid in enumerate(centroids):
                dist = np.sqrt((centroid[0] - text_center[0])**2 + 
                             (centroid[1] - text_center[1])**2)
                if dist < distances[i]:
                    distances[i] = dist
                    assignments[i] = elev['value']
        
        # Only assign if within max_distance
        for i in range(len(contours)):
            if distances[i] > max_distance:
                assignments[i] = 0
        
        return assignments
    
    def detect_elevations_from_contours(self, image, contours):
        """Complete pipeline: detect elevations and assign to contours"""
        elevations = self.detect_elevations(image)
        
        if not elevations:
            return [0] * len(contours)
        
        assignments = self.map_elevations_to_contours(contours, elevations)
        
        return assignments

# Test function
def test_ocr():
    """Test if OCR is working"""
    print("Testing OCR...")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        print("OCR is working!")
        return True
    except Exception as e:
        print(f"OCR not working: {e}")
        print("\nPlease install Tesseract:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install Tesseract 5.x")
        print("3. Make sure to select Greek language during installation")
        return False

if __name__ == "__main__":
    test_ocr()
