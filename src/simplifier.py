"""
Polyline simplification module
"""
import numpy as np
from scipy import interpolate
from scipy.spatial import distance

class PolylineSimplifier:
    """Simplify and smooth polylines"""
    
    def __init__(self):
        self.algorithm = 'douglas_peucker'
        self.epsilon = 1.0
        self.spline_smooth = 0.5
        self.min_points = 5
        
    def simplify(self, contour, method='douglas_peucker', epsilon=None):
        """Simplify contour using specified method"""
        if len(contour) < self.min_points:
            return contour
            
        if epsilon is None:
            epsilon = self.epsilon
            
        if method == 'douglas_peucker':
            return self.douglas_peucker(contour, epsilon)
        elif method == 'spline':
            return self.spline_simplify(contour)
        elif method == 'visvalingam':
            return self.visvalingam_whyatt(contour, epsilon)
        else:
            return contour
    
    def douglas_peucker(self, points, epsilon):
        """Douglas-Peucker simplification algorithm"""
        if len(points) <= 2:
            return points
            
        # Find point with maximum distance
        dmax = 0
        index = 0
        end = len(points) - 1
        
        for i in range(1, end):
            d = self._perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = self.douglas_peucker(points[:index+1], epsilon)
            rec_results2 = self.douglas_peucker(points[index:], epsilon)
            
            # Build result list
            result = rec_results1[:-1] + rec_results2
        else:
            result = [points[0], points[end]]
            
        return result
    
    def _perpendicular_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # If line is a point
        if x1 == x2 and y1 == y2:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Calculate perpendicular distance
        numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return numerator / denominator
    
    def spline_simplify(self, points, smooth=0.5):
        """Simplify using spline interpolation"""
        if len(points) < 4:
            return points
            
        points = np.array(points)
        
        # Remove duplicate points
        unique = [points[0]]
        for p in points[1:]:
            if not np.array_equal(p, unique[-1]):
                unique.append(p)
        points = np.array(unique)
        
        if len(points) < 4:
            return points.tolist()
        
        # Parameterize points
        t = np.zeros(len(points))
        for i in range(1, len(points)):
            t[i] = t[i-1] + np.sqrt(np.sum((points[i] - points[i-1])**2))
        
        # Normalize t
        t = t / t[-1]
        
        # Fit spline
        try:
            # Univariate spline
            from scipy.interpolate import UnivariateSpline
            
            # Fit x and y separately
            spline_x = UnivariateSpline(t, points[:, 0], s=smooth * len(points))
            spline_y = UnivariateSpline(t, points[:, 1], s=smooth * len(points))
            
            # Sample spline at more points
            t_new = np.linspace(0, 1, len(points) * 2)
            x_new = spline_x(t_new)
            y_new = spline_y(t_new)
            
            simplified = np.column_stack((x_new, y_new))
            
            # Douglas-Peucker to simplify further
            return self.douglas_peucker(simplified, 1.0)
            
        except Exception as e:
            print(f"Spline simplification failed: {e}")
            return points.tolist()
    
    def visvalingam_whyatt(self, points, threshold):
        """Visvalingam-Whyatt line simplification"""
        if len(points) <= 2:
            return points
            
        # Calculate area for each point
        areas = []
        for i in range(1, len(points) - 1):
            area = self._triangle_area(points[i-1], points[i], points[i+1])
            areas.append((i, area))
        
        # Sort by area
        areas.sort(key=lambda x: x[1])
        
        # Remove points with smallest area until threshold is met
        keep = [True] * len(points)
        removed = 0
        target = max(2, int(len(points) * 0.1))  # Keep at least 10%
        
        for idx, area in areas:
            if len([k for k in keep if k]) <= target:
                break
            if area < threshold:
                keep[idx] = False
                removed += 1
        
        # Build simplified points
        simplified = [p for i, p in enumerate(points) if keep[i]]
        
        return simplified
    
    def _triangle_area(self, p1, p2, p3):
        """Calculate area of triangle"""
        return abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2
    
    def smooth_contour(self, contour, window_size=3, iterations=1):
        """Smooth contour using moving average"""
        if len(contour) < 3:
            return contour
            
        points = np.array(contour)
        
        for _ in range(iterations):
            smoothed = points.copy()
            for i in range(len(points)):
                # Get window indices
                start = max(0, i - window_size)
                end = min(len(points), i + window_size + 1)
                window = points[start:end]
                
                # Average
                smoothed[i] = np.mean(window, axis=0)
            
            points = smoothed
        
        return points.tolist()
