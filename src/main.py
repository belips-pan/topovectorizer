"""
Main entry point for TopoVectorizer
"""
import sys
import argparse
import os

from gui import TopoVectorizerGUI
from reader import RasterReader
from preprocessing import Preprocessor
from skeleton import Skeletonizer
from graph import GraphAnalyzer
from tracker import ContourTracker
from simplifier import PolylineSimplifier
from georef import Georeferencer
from dxf_export import DXFExporter
from ocr import OCRDetector

def batch_process(input_dir, output_dir, format='dxf'):
    """Batch process multiple files"""
    # Initialize modules
    preprocessor = Preprocessor()
    skeletonizer = Skeletonizer()
    graph_analyzer = GraphAnalyzer()
    tracker = ContourTracker()
    simplifier = PolylineSimplifier()
    georeferencer = Georeferencer()
    dxf_exporter = DXFExporter()
    ocr_detector = OCRDetector()
    
    # Process all TIFF files
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.{format}")
            
            print(f"Processing: {filename}")
            
            try:
                # Read image
                reader = RasterReader(input_path)
                image = reader.read()
                
                if image is None:
                    print(f"  Error: Failed to read {filename}")
                    continue
                
                # Get georeferencing
                if reader.transform:
                    georeferencer.set_georeferencing(
                        reader.transform, reader.pixel_size, reader.crs
                    )
                
                # Process
                processed = preprocessor.process(image)
                skeleton = skeletonizer.skeletonize(processed)
                graph_analyzer.build_graph(skeleton)
                contours = tracker.track_contours(skeleton)
                contours = tracker.merge_contours(contours)
                
                # Simplify
                simplified = []
                for contour in contours:
                    simplified.append(simplifier.simplify(contour))
                
                # Detect elevations
                elevations = ocr_detector.detect_elevations(image)
                assignments = ocr_detector.map_elevations_to_contours(simplified, elevations)
                
                # Georeference
                if georeferencer.transform:
                    georeferenced = georeferencer.transform_contours(simplified)
                else:
                    georeferenced = simplified
                
                # Export
                if format.lower() == 'dxf':
                    dxf_exporter.export_contours(georeferenced, assignments, output_path)
                elif format.lower() == 'shp':
                    dxf_exporter.export_shapefile(georeferenced, assignments, output_path)
                elif format.lower() == 'gpkg':
                    dxf_exporter.export_geopackage(georeferenced, assignments, output_path)
                elif format.lower() == 'geojson':
                    dxf_exporter.export_geojson(georeferenced, assignments, output_path)
                
                print(f"  Output: {output_path}")
                print(f"  Contours: {len(simplified)}, Elevations detected: {len(assignments)}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='TopoVectorizer - Contour Vectorization Tool')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--input', type=str, help='Input directory for batch processing')
    parser.add_argument('--output', type=str, help='Output directory for batch processing')
    parser.add_argument('--format', type=str, default='dxf', 
                       choices=['dxf', 'shp', 'gpkg', 'geojson'],
                       help='Output format for batch processing')
    parser.add_argument('--gui', action='store_true', default=True,
                       help='Launch GUI (default)')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input or not args.output:
            print("Error: --input and --output required for batch processing")
            sys.exit(1)
        
        os.makedirs(args.output, exist_ok=True)
        batch_process(args.input, args.output, args.format)
    else:
        # Launch GUI
        from PySide6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        window = TopoVectorizerGUI()
        window.show()
        sys.exit(app.exec())

if __name__ == '__main__':
    main()
