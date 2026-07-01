"""
Module for reading various raster formats with memory optimization
"""
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
import os
import gc

class RasterReader:
    """Read TIFF and other raster formats with georeferencing"""
    
    def __init__(self, filepath, max_size=4096):
        self.filepath = filepath
        self.image = None
        self.transform = None
        self.crs = None
        self.pixel_size = None
        self.bounds = None
        self.max_size = max_size  # Μέγιστο μέγεθος για φόρτωση
        
    def read(self):
        """Read raster file with memory optimization"""
        try:
            # Try reading with rasterio
            with rasterio.open(self.filepath) as src:
                # Ελέγχουμε το μέγεθος
                height, width = src.height, src.width
                print(f"Original size: {height} x {width}")
                
                # Αν η εικόνα είναι πολύ μεγάλη, την κάνουμε resize
                if height > self.max_size or width > self.max_size:
                    scale = min(self.max_size / height, self.max_size / width)
                    new_height = int(height * scale)
                    new_width = int(width * scale)
                    print(f"Resizing to: {new_height} x {new_width}")
                    
                    # Διαβάζουμε με resampling
                    out_shape = (src.count, new_height, new_width)
                    self.image = src.read(
                        out_shape=out_shape,
                        resampling=Resampling.bilinear
                    )
                    
                    # Ενημερώνουμε το transform
                    self.transform = src.transform * src.transform.scale(
                        (width / new_width),
                        (height / new_height)
                    )
                else:
                    # Διαβάζουμε κανονικά
                    self.image = src.read(1)  # Μόνο το πρώτο κανάλι
                
                self.crs = src.crs
                self.pixel_size = (src.res[0], src.res[1])
                self.bounds = src.bounds
                
                # Αν έχουμε πολλά κανάλια, κρατάμε μόνο το πρώτο
                if len(self.image.shape) > 2:
                    self.image = self.image[0]  # Πρώτο κανάλι
                
                # Ελευθερώνουμε μνήμη
                gc.collect()
                
                return self.image
                
        except Exception as e:
            print(f"Rasterio read failed: {e}")
            # Fallback to PIL
            try:
                # Φορτώνουμε την εικόνα σε grayscale
                img = Image.open(self.filepath)
                
                # Μετατροπή σε grayscale
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('L')
                elif img.mode == 'P':
                    img = img.convert('L')
                elif img.mode == 'RGB':
                    img = img.convert('L')
                
                # Resize αν χρειάζεται
                if img.size[0] > self.max_size or img.size[1] > self.max_size:
                    img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
                
                self.image = np.array(img, dtype=np.uint8)
                
                # Ελευθερώνουμε μνήμη
                img.close()
                gc.collect()
                
                self._read_georeferencing()
                return self.image
                
            except Exception as e2:
                print(f"PIL read failed: {e2}")
                return None
    
    def read_optimized(self, target_size=2048):
        """Read image with aggressive memory optimization"""
        try:
            # Προσπαθούμε να διαβάσουμε μόνο ένα μέρος της εικόνας
            with rasterio.open(self.filepath) as src:
                height, width = src.height, src.width
                
                # Αν είναι πολύ μεγάλη, διαβάζουμε σε chunks
                if height > target_size or width > target_size:
                    scale = min(target_size / height, target_size / width)
                    new_height = int(height * scale)
                    new_width = int(width * scale)
                    
                    # Διαβάζουμε με resampling
                    out_shape = (1, new_height, new_width)
                    self.image = src.read(
                        1,  # Πρώτο κανάλι
                        out_shape=out_shape,
                        resampling=Resampling.bilinear
                    )
                    
                    self.transform = src.transform * src.transform.scale(
                        (width / new_width),
                        (height / new_height)
                    )
                else:
                    self.image = src.read(1)
                
                self.crs = src.crs
                self.pixel_size = (src.res[0], src.res[1])
                self.bounds = src.bounds
                
                # Εκκαθάριση μνήμης
                gc.collect()
                
                return self.image
                
        except Exception as e:
            print(f"Optimized read failed: {e}")
            return None
    
    def _read_georeferencing(self):
        """Read georeferencing from aux.xml or world file"""
        base = os.path.splitext(self.filepath)[0]
        
        # Try aux.xml
        aux_file = base + '.aux.xml'
        if os.path.exists(aux_file):
            self._parse_aux_xml(aux_file)
            return
            
        # Try world file
        for ext in ['.tfw', '.jgw', '.pgw', '.gfw']:
            world_file = base + ext
            if os.path.exists(world_file):
                self._parse_world_file(world_file)
                return
    
    def _parse_aux_xml(self, aux_file):
        """Parse ESRI aux.xml file"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(aux_file)
            root = tree.getroot()
            
            for elem in root.iter():
                if 'GeoTransform' in elem.tag:
                    values = elem.text.split()
                    if len(values) >= 6:
                        self.transform = tuple(map(float, values))
                        self.pixel_size = (abs(float(values[1])), abs(float(values[5])))
                        break
                        
            for elem in root.iter():
                if 'SpatialReference' in elem.tag:
                    self.crs = elem.text
                    break
                    
        except Exception as e:
            print(f"Error parsing aux.xml: {e}")
    
    def _parse_world_file(self, world_file):
        """Parse world file (.tfw, .jgw, etc.)"""
        try:
            with open(world_file, 'r') as f:
                lines = [float(line.strip()) for line in f.readlines() if line.strip()]
                
            if len(lines) >= 6:
                self.pixel_size = (abs(lines[0]), abs(lines[3]))
                self.transform = (lines[4], lines[0], lines[1], 
                                 lines[5], lines[2], lines[3])
        except Exception as e:
            print(f"Error parsing world file: {e}")
    
    def pixel_to_coord(self, row, col):
        """Convert pixel coordinates to geographic coordinates"""
        if self.transform is None:
            return None
            
        x = self.transform[0] + col * self.transform[1] + row * self.transform[2]
        y = self.transform[3] + col * self.transform[4] + row * self.transform[5]
        return (x, y)
    
    def coord_to_pixel(self, x, y):
        """Convert geographic coordinates to pixel coordinates"""
        if self.transform is None:
            return None
            
        det = self.transform[1] * self.transform[5] - self.transform[2] * self.transform[4]
        if abs(det) < 1e-10:
            return None
            
        col = (self.transform[5] * (x - self.transform[0]) - 
               self.transform[2] * (y - self.transform[3])) / det
        row = (-self.transform[4] * (x - self.transform[0]) + 
               self.transform[1] * (y - self.transform[3])) / det
        return (int(row), int(col))
