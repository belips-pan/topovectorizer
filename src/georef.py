"""
Georeferencing module for coordinate transformation
"""
import numpy as np
import pyproj
import json

class Georeferencer:
    """Handle georeferencing and coordinate transformations"""
    
    def __init__(self):
        self.transform = None
        self.pixel_size = None
        self.source_crs = None
        self.target_crs = 'EPSG:2100'  # EGSA87
        self.proj_transformer = None
        
    def set_georeferencing(self, transform, pixel_size, crs=None):
        """Set georeferencing parameters"""
        self.transform = transform
        self.pixel_size = pixel_size
        self.source_crs = crs
        
        # Setup coordinate transformer
        if crs:
            try:
                self.proj_transformer = pyproj.Transformer.from_crs(
                    crs, self.target_crs, always_xy=True
                )
            except Exception as e:
                print(f"Warning: Could not create transformer for {crs}: {e}")
                self.proj_transformer = None
    
    def pixel_to_coord(self, row, col):
        """Convert pixel coordinates to geographic coordinates"""
        if self.transform is None:
            return None
            
        # Using affine transform
        x = self.transform[0] + col * self.transform[1] + row * self.transform[2]
        y = self.transform[3] + col * self.transform[4] + row * self.transform[5]
        
        # Transform to target CRS if needed
        if self.proj_transformer:
            try:
                x, y = self.proj_transformer.transform(x, y)
            except:
                pass
        
        return (x, y)
    
    def coord_to_pixel(self, x, y):
        """Convert geographic coordinates to pixel coordinates"""
        if self.transform is None:
            return None
            
        # Inverse affine transform
        det = self.transform[1] * self.transform[5] - self.transform[2] * self.transform[4]
        if abs(det) < 1e-10:
            return None
            
        # Transform from target to source CRS if needed
        if self.proj_transformer:
            try:
                transformer_inv = pyproj.Transformer.from_crs(
                    self.target_crs, self.source_crs, always_xy=True
                )
                x, y = transformer_inv.transform(x, y)
            except:
                pass
        
        col = (self.transform[5] * (x - self.transform[0]) - 
               self.transform[2] * (y - self.transform[3])) / det
        row = (-self.transform[4] * (x - self.transform[0]) + 
               self.transform[1] * (y - self.transform[3])) / det
        return (int(row), int(col))
    
    def transform_contours(self, contours, flip_y=True, image_height=None):
        """Transform all contours to geographic coordinates with optional Y flip"""
        if self.transform is None:
            print("Warning: No transform available, returning original contours")
            return contours
            
        transformed = []
        for contour in contours:
            geo_contour = []
            for y, x in contour:
                # Αν χρειάζεται, αντιστρέφουμε τον Y άξονα
                if flip_y and image_height is not None:
                    y = image_height - y
                
                coord = self.pixel_to_coord(y, x)
                if coord:
                    geo_contour.append(coord)
            if len(geo_contour) > 1:
                transformed.append(geo_contour)
         
        print(f"Transformed {len(transformed)} contours to geographic coordinates")
        return transformed
    
    def export_world_file(self, filepath):
        """Export world file for the current georeferencing"""
        if self.transform is None:
            return
            
        with open(filepath, 'w') as f:
            f.write(f"{self.transform[1]:.12f}\n")
            f.write(f"{self.transform[2]:.12f}\n")
            f.write(f"{self.transform[4]:.12f}\n")
            f.write(f"{self.transform[5]:.12f}\n")
            f.write(f"{self.transform[0]:.12f}\n")
            f.write(f"{self.transform[3]:.12f}\n")
    
    def export_geojson(self, contours, output_file, properties=None):
        """Export contours as GeoJSON"""
        import json
        
        features = []
        for i, contour in enumerate(contours):
            # Close the contour if it's a loop
            coords = contour
            if len(contour) > 0 and not np.array_equal(contour[0], contour[-1]):
                coords = contour + [contour[0]]
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[x, y] for y, x in coords]
                },
                "properties": properties[i] if properties and i < len(properties) else {"id": i}
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
