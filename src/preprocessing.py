"""
Image preprocessing module for contour extraction
"""
import numpy as np
import cv2
import gc

class Preprocessor:
    """Preprocess raster images for contour extraction"""
    
    def __init__(self):
        self.threshold_value = 128
        
    def process(self, image, max_size=4096):
        """Complete preprocessing pipeline"""
        print(f"Original image shape: {image.shape}, dtype: {image.dtype}")
        
        # Resize if too large
        if image.shape[0] > max_size or image.shape[1] > max_size:
            scale = min(max_size / image.shape[0], max_size / image.shape[1])
            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized to: {image.shape}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Converted to grayscale")
        
        # Normalize if needed
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        print(f"Image stats - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}")
        
        # ΕΛΕΓΧΟΣ: Η εικόνα έχει λευκές γραμμές σε μαύρο φόντο
        # Οι ισοϋψείς είναι οι λευκές γραμμές (τιμές > 200)
        # Το φόντο είναι μαύρο (τιμές < 50)
        
        # Βήμα 1: Ενίσχυση αντίθεσης
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Βήμα 2: Βρίσκουμε τις λευκές γραμμές με απλό threshold
        # Οι ισοϋψείς είναι οι πιο φωτεινές περιοχές
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        percentage = (white_pixels / total_pixels) * 100
        
        print(f"Threshold 200: {white_pixels} white pixels ({percentage:.2f}%)")
        
        # Αν δεν βρήκε αρκετές γραμμές, δοκιμάζουμε χαμηλότερο threshold
        if white_pixels < 1000:
            print("Too few pixels, trying lower threshold...")
            _, binary = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
            white_pixels = np.sum(binary > 0)
            percentage = (white_pixels / total_pixels) * 100
            print(f"Threshold 150: {white_pixels} white pixels ({percentage:.2f}%)")
        
        # Αν πάλι δεν βρήκε, δοκιμάζουμε adaptive threshold
        if white_pixels < 1000:
            print("Still too few, trying adaptive threshold...")
            binary = cv2.adaptiveThreshold(enhanced, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 5)
            white_pixels = np.sum(binary > 0)
            percentage = (white_pixels / total_pixels) * 100
            print(f"Adaptive: {white_pixels} white pixels ({percentage:.2f}%)")
        
        # Αν έχει πάρα πολλά pixels (> 30%), ίσως οι ισοϋψείς είναι μαύρες
        if percentage > 30:
            print("Too many white pixels, inverting...")
            binary = cv2.bitwise_not(binary)
            white_pixels = np.sum(binary > 0)
            percentage = (white_pixels / total_pixels) * 100
            print(f"Inverted: {white_pixels} white pixels ({percentage:.2f}%)")
        
        print(f"Final white pixels: {white_pixels} ({percentage:.2f}%)")
        
        # Βήμα 3: Καθαρισμός
        # Αφαίρεση μικρού θορύβου
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Σύνδεση κοντινών γραμμών
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Αποθήκευση για έλεγχο
        cv2.imwrite("debug_preprocess.png", binary)
        print("Saved debug_preprocess.png for inspection")
        
        gc.collect()
        return binary
