"""
Skeletonization module for thinning contour lines to single pixel width
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize as skimage_skeletonize
import cv2
import gc  # Προσθήκη

class Skeletonizer:
    """Apply skeletonization to binary images"""
    
    def __init__(self, method='zhang_suen'):
        self.method = method
        self.max_iterations = 100
        self.preserve_endpoints = True
        
    def skeletonize(self, image):
        """Apply skeletonization to binary image"""
        # Ensure binary image
        if image.dtype != np.bool:
            binary = image > 0
        else:
            binary = image
        
        # Διασφάλιση ότι η εικόνα έχει επαρκές πάχος για σκελετοποίηση
        if np.sum(binary) < 1000:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1)
            binary = binary > 0
        
        print(f"Skeletonizing image with {np.sum(binary)} foreground pixels")
        
        # Εφαρμογή σκελετοποίησης
        if self.method == 'zhang_suen':
            skeleton = self.zhang_suen_skeleton(binary)
        elif self.method == 'lee':
            skeleton = self.lee_skeleton(binary)
        else:
            skeleton = skimage_skeletonize(binary)
        
        # Clean up skeleton
        skeleton = self.clean_skeleton(skeleton)
        
        # Έλεγχος αν το skeleton είναι άδειο
        if not np.any(skeleton):
            print("Warning: Skeleton is empty, returning binary image")
            return binary.astype(np.uint8) * 255
        
        print(f"Skeleton has {np.sum(skeleton)} pixels")
        
        # Εκκαθάριση μνήμης
        gc.collect()
        
        return skeleton.astype(np.uint8) * 255
    
    def zhang_suen_skeleton(self, img):
        """Zhang-Suen thinning algorithm"""
        img = img.astype(np.uint8)
        padded = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        current = padded.copy()
        
        changed = True
        iteration = 0
        
        while changed and iteration < self.max_iterations:
            changed = False
            marker = np.zeros_like(current, dtype=bool)
            
            # Sub-iteration 1
            for i in range(1, current.shape[0]-1):
                for j in range(1, current.shape[1]-1):
                    if current[i, j] == 0:
                        continue
                    
                    p2, p3, p4 = current[i-1, j], current[i-1, j+1], current[i, j+1]
                    p5, p6, p7 = current[i+1, j+1], current[i+1, j], current[i+1, j-1]
                    p8, p9 = current[i, j-1], current[i-1, j-1]
                    
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if not (2 <= B <= 6):
                        continue
                    
                    A = sum([
                        1 if p2 == 0 and p3 == 1 else 0,
                        1 if p3 == 0 and p4 == 1 else 0,
                        1 if p4 == 0 and p5 == 1 else 0,
                        1 if p5 == 0 and p6 == 1 else 0,
                        1 if p6 == 0 and p7 == 1 else 0,
                        1 if p7 == 0 and p8 == 1 else 0,
                        1 if p8 == 0 and p9 == 1 else 0,
                        1 if p9 == 0 and p2 == 1 else 0
                    ])
                    if A != 1:
                        continue
                    
                    if p2 * p4 * p6 != 0:
                        continue
                    
                    if p4 * p6 * p8 != 0:
                        continue
                    
                    marker[i, j] = True
                    changed = True
            
            current[marker] = 0
            marker.fill(False)
            
            # Sub-iteration 2
            for i in range(1, current.shape[0]-1):
                for j in range(1, current.shape[1]-1):
                    if current[i, j] == 0:
                        continue
                    
                    p2, p3, p4 = current[i-1, j], current[i-1, j+1], current[i, j+1]
                    p5, p6, p7 = current[i+1, j+1], current[i+1, j], current[i+1, j-1]
                    p8, p9 = current[i, j-1], current[i-1, j-1]
                    
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if not (2 <= B <= 6):
                        continue
                    
                    A = sum([
                        1 if p2 == 0 and p3 == 1 else 0,
                        1 if p3 == 0 and p4 == 1 else 0,
                        1 if p4 == 0 and p5 == 1 else 0,
                        1 if p5 == 0 and p6 == 1 else 0,
                        1 if p6 == 0 and p7 == 1 else 0,
                        1 if p7 == 0 and p8 == 1 else 0,
                        1 if p8 == 0 and p9 == 1 else 0,
                        1 if p9 == 0 and p2 == 1 else 0
                    ])
                    if A != 1:
                        continue
                    
                    if p2 * p4 * p8 != 0:
                        continue
                    
                    if p2 * p6 * p8 != 0:
                        continue
                    
                    marker[i, j] = True
                    changed = True
            
            current[marker] = 0
            iteration += 1
        
        skeleton = current[1:-1, 1:-1]
        return skeleton
    
    def lee_skeleton(self, img, iterations=-1):
        """Lee skeletonization algorithm"""
        from skimage.morphology import skeletonize as lee_skeleton
        
        if iterations == -1:
            return lee_skeleton(img, method='lee')
        else:
            skeleton = img.copy()
            for _ in range(iterations):
                prev = skeleton.copy()
                skeleton = lee_skeleton(skeleton, method='lee')
                if np.array_equal(skeleton, prev):
                    break
            return skeleton
    
    def clean_skeleton(self, skeleton):
        """Clean up skeleton"""
        cleaned = ndimage.binary_erosion(skeleton, structure=np.ones((3, 3)))
        cleaned = ndimage.binary_dilation(cleaned, structure=np.ones((3, 3)))
        
        from skimage.morphology import remove_small_objects
        cleaned = remove_small_objects(cleaned, min_size=20)
        
        return cleaned
