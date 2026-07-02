"""
DXF Export Module with proper coordinate handling
"""
import ezdxf
from ezdxf.math import Vec3
import numpy as np

class DXFExporter:
    """Export contours to DXF format"""
    
    def __init__(self):
        self.doc = None
        self.modelspace = None
        self.layer_name = "Contours"
        self.color = 7  # White
        self.use_3d = False
        self.image_height = None
        self.has_georeferencing = False  # Νέο flag
        
    def create_dxf(self, filename="output.dxf"):
        """Create new DXF document"""
        self.doc = ezdxf.new()
        self.modelspace = self.doc.modelspace()
        self.doc.layers.add(self.layer_name, color=self.color)
        
    def export_contours(self, contours, elevations=None, filename="output.dxf", 
                       image_height=None, has_georeferencing=False):
        """
        Export contours to DXF with proper coordinate handling
        
        Args:
            contours: List of contours (pixel coordinates)
            elevations: List of elevation values
            filename: Output DXF file path
            image_height: Height of image for Y correction
            has_georeferencing: True if georeferencing was applied
        """
        self.image_height = image_height
        self.has_georeferencing = has_georeferencing
        
        # Δημιουργούμε το DXF
        self.create_dxf(filename)
        
        if elevations is None:
            elevations = [0] * len(contours)
        
        exported_count = 0
        
        for contour, elev in zip(contours, elevations):
            if len(contour) < 2:
                continue
            
            # --- ΕΠΕΞΕΡΓΑΣΙΑ ΣΥΝΤΕΤΑΓΜΕΝΩΝ ---
            if has_georeferencing:
                # Αν έχουμε γεωαναφορά, οι συντεταγμένες είναι ήδη σε ΕΓΣΑ87
                # Δεν κάνουμε καμία αλλαγή
                corrected_contour = contour
                self.log_message("Using georeferenced coordinates (EGSA87)")
            else:
                # ΧΩΡΙΣ γεωαναφορά: Χρησιμοποιούμε τις pixel συντεταγμένες
                # Αλλά διατηρούμε την αναλογία και τη θέση της εικόνας
                corrected_contour = self._convert_pixel_to_dxf(contour)
            
            if len(corrected_contour) < 2:
                continue
            
            # Δημιουργία polyline
            if self.use_3d and elev != 0:
                # 3D Polyline
                points = [Vec3(x, y, elev) for y, x in corrected_contour]
                self.modelspace.add_polyline3d(points, dxfattribs={
                    'layer': self.layer_name,
                    'color': self.color
                })
            else:
                # 2D Polyline
                points = [(x, y) for y, x in corrected_contour]
                if len(points) > 2 and self._is_closed(points):
                    self.modelspace.add_lwpolyline(points, close=True, dxfattribs={
                        'layer': self.layer_name,
                        'color': self.color
                    })
                else:
                    self.modelspace.add_lwpolyline(points, dxfattribs={
                        'layer': self.layer_name,
                        'color': self.color
                    })
            
            exported_count += 1
        
        self.doc.saveas(filename)
        print(f"DXF exported: {filename}")
        print(f"Exported {exported_count} contours")
        if not has_georeferencing:
            print("Note: Using pixel coordinates (no georeferencing)")
        return exported_count
    
    def _convert_pixel_to_dxf(self, contour):
        """
        Μετατροπή pixel coordinates σε DXF coordinates
        Διατηρεί τη σχετική θέση των σημείων
        """
        if not contour:
            return contour
        
        # Βρίσκουμε τα όρια της εικόνας
        if self.image_height is not None:
            # Χρησιμοποιούμε το γνωστό ύψος
            height = self.image_height
        else:
            # Βρίσκουμε το max Y από τα σημεία
            height = max([y for y, x in contour]) + 1
        
        # Αντιστρέφουμε τον Y άξονα (pixel → AutoCAD)
        # Αλλά διατηρούμε τις σχετικές αποστάσεις
        corrected = []
        for y, x in contour:
            # Το AutoCAD έχει τον Y άξονα από κάτω προς τα πάνω
            # Ενώ τα pixels είναι από πάνω προς τα κάτω
            new_y = height - y
            corrected.append((new_y, x))
        
        return corrected
    
    def _is_closed(self, points):
        """Check if polyline should be closed"""
        if len(points) < 3:
            return False
        
        # Έλεγχος αν το πρώτο και τελευταίο σημείο είναι κοντά
        p1 = points[0]
        p2 = points[-1]
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        return distance < 1.0
    
    def log_message(self, msg):
        """Print message (for debugging)"""
        print(msg)
          
