"""
Polyline tracking module for following contours
"""
import numpy as np
import cv2
from scipy import ndimage
from collections import deque

class ContourTracker:
    """Track and extract contours from skeleton"""
    
    def __init__(self):
        self.contours = []
        self.visited = None
        self.skeleton = None
        self.use_cv2 = False  # Εναλλαγή μεθόδου
        
    def track_contours(self, skeleton, use_cv2=False):
        """Track all contours in skeleton"""
        self.skeleton = skeleton > 0 if skeleton is not None else np.array([[]])
        self.visited = np.zeros_like(self.skeleton, dtype=bool)
        self.contours = []
        self.use_cv2 = use_cv2
        
        # Έλεγχος αν το skeleton είναι άδειο
        if self.skeleton.size == 0 or not np.any(self.skeleton):
            print("Warning: Empty skeleton, no contours to track")
            return []
        
        # Αν η εικόνα είναι πολύ μεγάλη ή ο σκελετός έχει λίγα pixels, χρησιμοποιούμε OpenCV
        total_pixels = np.sum(self.skeleton)
        print(f"Skeleton has {total_pixels} pixels")
        
        # Αν ο σκελετός έχει πολύ λίγα pixels ή έχει σπάσει, χρησιμοποιούμε OpenCV
        if use_cv2 or total_pixels < 1000:
            print("Using OpenCV contour extraction (fallback method)")
            return self._track_contours_cv2()
        else:
            print("Using graph-based contour extraction")
            return self._track_contours_graph()
    
    def _track_contours_graph(self):
        """Track contours using graph analysis"""
        # Find all starting points (endpoints or any unvisited pixel)
        start_points = self._find_start_points()
        
        if not start_points:
            print("No starting points found, using OpenCV fallback")
            return self._track_contours_cv2()
        
        for y, x in start_points:
            if 0 <= y < self.visited.shape[0] and 0 <= x < self.visited.shape[1]:
                if not self.visited[y, x]:
                    contour = self._trace_contour(y, x)
                    if len(contour) > 10:  # Minimum contour length
                        self.contours.append(contour)
        
        # Sort contours by length (descending)
        self.contours.sort(key=len, reverse=True)
        
        print(f"Extracted {len(self.contours)} contours using graph method")
        return self.contours
    
    def _track_contours_cv2(self):
        """Track contours using OpenCV (more robust for broken skeletons)"""
        # Convert to uint8
        skel_uint8 = (self.skeleton > 0).astype(np.uint8) * 255
        
        # Εφαρμογή dilation για να ενώσουμε σπασμένες γραμμές
        kernel = np.ones((3, 3), np.uint8)
        skel_dilated = cv2.dilate(skel_uint8, kernel, iterations=2)
        
        # Εύρεση contours
        contours, hierarchy = cv2.findContours(skel_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert to point lists and filter
        self.contours = []
        for contour in contours:
            if len(contour) > 10:
                # Συνέχεια: κάνουμε smoothing των points
                points = [(int(p[0][1]), int(p[0][0])) for p in contour]
                
                # Αφαίρεση διπλών σημείων
                unique_points = [points[0]]
                for p in points[1:]:
                    if p != unique_points[-1]:
                        unique_points.append(p)
                
                if len(unique_points) > 10:
                    self.contours.append(unique_points)
        
        # Συγχώνευση κοντινών contours
        self.contours = self.merge_contours(self.contours, max_gap=10)
        
        print(f"Extracted {len(self.contours)} contours using OpenCV method")
        return self.contours
    
    def _find_start_points(self):
        """Find starting points for contour tracking"""
        start_points = []
        
        if self.skeleton is None or not np.any(self.skeleton):
            return start_points
        
        # Find endpoints (degree 1)
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        try:
            neighbors = ndimage.convolve(self.skeleton.astype(np.uint8), 
                                        kernel, mode='constant', cval=0)
            
            endpoints = np.argwhere(self.skeleton & (neighbors == 1))
            for y, x in endpoints:
                start_points.append((y, x))
        except Exception as e:
            print(f"Warning: Error finding endpoints: {e}")
        
        # If no endpoints found, use any skeleton pixel
        if len(start_points) == 0:
            pixels = np.argwhere(self.skeleton)
            if len(pixels) > 0:
                # Πάρτε μερικά τυχαία σημεία
                indices = np.random.choice(len(pixels), min(100, len(pixels)), replace=False)
                for idx in indices:
                    start_points.append(tuple(pixels[idx]))
        
        return start_points
    
    def _trace_contour(self, start_y, start_x):
        """Trace a single contour from starting point"""
        contour = []
        current = (start_y, start_x)
        prev = None
        
        max_steps = 50000  # Αποτροπή ατέρμονου βρόχου
        steps = 0
        
        while steps < max_steps:
            steps += 1
            
            if not (0 <= current[0] < self.visited.shape[0] and 0 <= current[1] < self.visited.shape[1]):
                break
                
            if self.visited[current[0], current[1]]:
                break
                
            contour.append(current)
            self.visited[current[0], current[1]] = True
            
            # Find next pixel (8-neighborhood)
            next_pos = self._find_next_pixel(current[0], current[1], prev)
            
            if next_pos is None:
                break
                
            prev = current
            current = next_pos
            
            # Check if we've reached an endpoint
            if self._is_endpoint(current[0], current[1]):
                contour.append(current)
                self.visited[current[0], current[1]] = True
                break
        
        return contour
    
    def _find_next_pixel(self, y, x, prev):
        """Find next pixel in contour"""
        # Check 8-neighborhood
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                    
                ny, nx_pos = y + dy, x + dx
                
                if not (0 <= ny < self.skeleton.shape[0] and 0 <= nx_pos < self.skeleton.shape[1]):
                    continue
                    
                if not self.skeleton[ny, nx_pos]:
                    continue
                    
                if self.visited[ny, nx_pos]:
                    continue
                    
                # Don't go back to previous pixel
                if prev is not None and (ny, nx_pos) == prev:
                    continue
                    
                # Prioritize pixels that maintain direction
                return (ny, nx_pos)
        
        return None
    
    def _is_endpoint(self, y, x):
        """Check if pixel is an endpoint"""
        if not (0 <= y < self.skeleton.shape[0] and 0 <= x < self.skeleton.shape[1]):
            return False
            
        if not self.skeleton[y, x]:
            return False
            
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_pos = y + dy, x + dx
                if 0 <= ny < self.skeleton.shape[0] and 0 <= nx_pos < self.skeleton.shape[1]:
                    if self.skeleton[ny, nx_pos]:
                        count += 1
        
        return count <= 1
    
    def merge_contours(self, contours, max_gap=10):
        """Merge contours that are close to each other"""
        if not contours or len(contours) <= 1:
            return contours
        
        print(f"Merging {len(contours)} contours with max_gap={max_gap}")
        
        merged = []
        used = [False] * len(contours)
        
        for i in range(len(contours)):
            if used[i]:
                continue
                
            current = contours[i]
            used[i] = True
            
            # Try to merge with other contours
            for j in range(i + 1, len(contours)):
                if used[j]:
                    continue
                    
                if self._can_merge(current, contours[j], max_gap):
                    current = self._merge_two_contours(current, contours[j])
                    used[j] = True
            
            merged.append(current)
        
        print(f"Merged to {len(merged)} contours")
        return merged
    
    def _can_merge(self, contour1, contour2, max_gap):
        """Check if two contours can be merged"""
        if not contour1 or not contour2:
            return False
            
        # Check distance between endpoints
        endpoints1 = [contour1[0], contour1[-1]]
        endpoints2 = [contour2[0], contour2[-1]]
        
        for e1 in endpoints1:
            for e2 in endpoints2:
                dist = np.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2)
                if dist <= max_gap:
                    return True
        
        return False
    
    def _merge_two_contours(self, contour1, contour2):
        """Merge two contours"""
        if not contour1 or not contour2:
            return contour1 if contour1 else contour2
            
        # Check which endpoints are closest
        endpoints1 = [contour1[0], contour1[-1]]
        endpoints2 = [contour2[0], contour2[-1]]
        
        min_dist = float('inf')
        best_pair = None
        
        for i, e1 in enumerate(endpoints1):
            for j, e2 in enumerate(endpoints2):
                dist = np.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)
        
        if best_pair is None:
            return contour1 + contour2
            
        # Merge based on closest endpoints
        if best_pair[0] == 0:  # contour1 start
            merged = contour1[::-1] + contour2
        else:  # contour1 end
            merged = contour1 + contour2
            
        return merged
    
    def get_contour_statistics(self):
        """Get statistics about extracted contours"""
        if not self.contours:
            return {'num_contours': 0}
            
        lengths = [len(c) for c in self.contours]
        stats = {
            'num_contours': len(self.contours),
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_length': np.mean(lengths) if lengths else 0,
            'total_points': sum(lengths) if lengths else 0
        }
        return stats
    
