"""
DXF Export Module
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
        
    def create_dxf(self, filename="output.dxf"):
        """Create new DXF document"""
        self.doc = ezdxf.new()
        self.modelspace = self.doc.modelspace()
        
        # Create layer
        self.doc.layers.add(self.layer_name, color=self.color)
        
    def export_contours(self, contours, elevations=None, filename="output.dxf"):
        """Export contours to DXF"""
        self.create_dxf(filename)
        
        if elevations is None:
            elevations = [0] * len(contours)
        
        for contour, elev in zip(contours, elevations):
            if len(contour) < 2:
                continue
            
            if self.use_3d and elev != 0:
                # Create 3D polyline
                points = [Vec3(x, y, elev) for y, x in contour]
                self.modelspace.add_polyline3d(points, dxfattribs={
                    'layer': self.layer_name,
                    'color': self.color
                })
            else:
                # Create 2D polyline
                points = [(x, y) for y, x in contour]
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
        
        self.doc.saveas(filename)
        
    def _is_closed(self, points):
        """Check if polyline should be closed"""
        if len(points) < 3:
            return False
        
        # Check if first and last points are close
        p1 = points[0]
        p2 = points[-1]
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # If distance < 1 pixel, consider it closed
        return distance < 1.0
    
    def export_shapefile(self, contours, elevations, filename="output.shp"):
        """Export to Shapefile using GeoPandas"""
        try:
            import geopandas as gpd
            from shapely.geometry import LineString
            import pandas as pd
            
            geometries = []
            for contour in contours:
                if len(contour) < 2:
                    continue
                # Convert to shapely LineString
                coords = [(x, y) for y, x in contour]
                geometries.append(LineString(coords))
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'elevation': elevations[:len(geometries)]
            })
            
            # Set CRS
            gdf.crs = "EPSG:2100"  # EGSA87
            
            # Save to shapefile
            gdf.to_file(filename, driver='ESRI Shapefile')
            
        except ImportError:
            print("GeoPandas not installed. Cannot export Shapefile.")
    
    def export_geopackage(self, contours, elevations, filename="output.gpkg"):
        """Export to GeoPackage"""
        try:
            import geopandas as gpd
            from shapely.geometry import LineString
            
            geometries = []
            for contour in contours:
                if len(contour) < 2:
                    continue
                coords = [(x, y) for y, x in contour]
                geometries.append(LineString(coords))
            
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'elevation': elevations[:len(geometries)]
            })
            gdf.crs = "EPSG:2100"
            gdf.to_file(filename, driver='GPKG')
            
        except ImportError:
            print("GeoPandas not installed. Cannot export GeoPackage.")
    
    def export_geojson(self, contours, elevations, filename="output.geojson"):
        """Export to GeoJSON"""
        try:
            from georef import Georeferencer
            georef = Georeferencer()
            # Just use the GeoJSON export from georef
            properties = [{"elevation": elev} for elev in elevations]
            georef.export_geojson(contours, filename, properties)
            
        except:
            import json
            features = []
            for contour, elev in zip(contours, elevations):
                if len(contour) < 2:
                    continue
                coords = [(x, y) for y, x in contour]
                if not np.array_equal(contour[0], contour[-1]):
                    coords.append(coords[0])
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {"elevation": elev}
                }
                features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            with open(filename, 'w') as f:
                json.dump(geojson, f, indent=2)
