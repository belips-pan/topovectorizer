"""
PySide6 GUI for TopoVectorizer - Enhanced Version
"""
import sys
import os
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import cv2

from reader import RasterReader
from preprocessing import Preprocessor
from skeleton import Skeletonizer
from graph import GraphAnalyzer
from tracker import ContourTracker
from simplifier import PolylineSimplifier
from georef import Georeferencer
from dxf_export import DXFExporter
from ocr import OCRDetector

class TopoVectorizerGUI(QMainWindow):
    """Main GUI window for TopoVectorizer"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Initialize modules
        self.reader = None
        self.preprocessor = Preprocessor()
        self.skeletonizer = Skeletonizer()
        self.graph_analyzer = GraphAnalyzer()
        self.tracker = ContourTracker()
        self.simplifier = PolylineSimplifier()
        self.georeferencer = Georeferencer()
        self.dxf_exporter = DXFExporter()
        self.ocr_detector = OCRDetector()
        
        # Data
        self.image = None
        self.processed_image = None
        self.skeleton = None
        self.contours = None
        self.simplified_contours = None
        self.elevations = None
        self.georeferenced = None
        self.current_file = None
        
        self.setWindowTitle("TopoVectorizer - Contour Vectorization")
        self.setGeometry(100, 100, 1400, 800)
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left side - Controls
        controls_widget = QWidget()
        controls_widget.setFixedWidth(380)
        controls_layout = QVBoxLayout(controls_widget)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("1. Load TIFF")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)
        
        # ΝΕΟ: Clear Project button
        self.clear_btn = QPushButton("🗑️ Clear Project")
        self.clear_btn.clicked.connect(self.clear_project)
        self.clear_btn.setStyleSheet("background-color: #ff6b6b; color: white;")
        file_layout.addWidget(self.clear_btn)
        
        file_layout.addWidget(QLabel(""))
        
        self.save_btn = QPushButton("Export DXF")
        self.save_btn.clicked.connect(self.export_dxf)
        file_layout.addWidget(self.save_btn)
        
        file_group.setLayout(file_layout)
        controls_layout.addWidget(file_group)
        
        # Processing controls
        process_group = QGroupBox("Processing Steps")
        process_layout = QVBoxLayout()
        
        self.preprocess_btn = QPushButton("2. Preprocess")
        self.preprocess_btn.clicked.connect(self.preprocess_image)
        process_layout.addWidget(self.preprocess_btn)
        
        self.skeleton_btn = QPushButton("3. Skeletonize")
        self.skeleton_btn.clicked.connect(self.skeletonize_image)
        process_layout.addWidget(self.skeleton_btn)
        
        # ΝΕΟ: Skip Skeleton button
        self.skip_skeleton_btn = QPushButton("⏭️ Skip Skeleton (Direct Contours)")
        self.skip_skeleton_btn.clicked.connect(self.skip_to_contours)
        self.skip_skeleton_btn.setStyleSheet("background-color: #ffd93d; color: black;")
        process_layout.addWidget(self.skip_skeleton_btn)
        
        self.analyze_btn = QPushButton("4. Extract Contours")
        self.analyze_btn.clicked.connect(self.extract_contours)
        process_layout.addWidget(self.analyze_btn)
        
        self.simplify_btn = QPushButton("5. Simplify Polylines")
        self.simplify_btn.clicked.connect(self.simplify_polylines)
        process_layout.addWidget(self.simplify_btn)
        
        self.ocr_btn = QPushButton("6. Detect Elevations (OCR)")
        self.ocr_btn.clicked.connect(self.detect_elevations)
        process_layout.addWidget(self.ocr_btn)
        
        self.georef_btn = QPushButton("7. Georeference")
        self.georef_btn.clicked.connect(self.georeference_contours)
        process_layout.addWidget(self.georef_btn)
        
        process_group.setLayout(process_layout)
        controls_layout.addWidget(process_group)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Threshold:"), 0, 0)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(128)
        params_layout.addWidget(self.threshold_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Epsilon:"), 1, 0)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.1, 10.0)
        self.epsilon_spin.setValue(1.0)
        params_layout.addWidget(self.epsilon_spin, 1, 1)
        
        params_layout.addWidget(QLabel("OCR Confidence:"), 2, 0)
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(60)
        params_layout.addWidget(self.confidence_spin, 2, 1)
        
        params_layout.addWidget(QLabel("Extraction Method:"), 3, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Auto", "Graph", "OpenCV"])
        self.method_combo.setCurrentIndex(0)
        params_layout.addWidget(self.method_combo, 3, 1)
        
        params_group.setLayout(params_layout)
        controls_layout.addWidget(params_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)
        controls_layout.addWidget(status_group)
        
        # ΝΕΟ: Exit button
        exit_layout = QHBoxLayout()
        exit_layout.addStretch()
        self.exit_btn = QPushButton("❌ Exit")
        self.exit_btn.clicked.connect(self.close)
        self.exit_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
        exit_layout.addWidget(self.exit_btn)
        controls_layout.addLayout(exit_layout)
        
        controls_layout.addStretch()
        
        # Right side - Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(viz_widget)
        
        main_layout.setStretchFactor(controls_widget, 0)
        main_layout.setStretchFactor(viz_widget, 1)
        
        # Συνδέσεις παραμέτρων
        self.threshold_spin.valueChanged.connect(self.update_parameters)
        self.epsilon_spin.valueChanged.connect(self.update_parameters)
        self.confidence_spin.valueChanged.connect(self.update_parameters)
        
        # Όλα τα πλήκτρα ενεργά
        self.set_all_buttons_enabled(True)
    
    def set_all_buttons_enabled(self, enabled):
        """Enable or disable all processing buttons"""
        self.preprocess_btn.setEnabled(enabled)
        self.skeleton_btn.setEnabled(enabled)
        self.skip_skeleton_btn.setEnabled(enabled)
        self.analyze_btn.setEnabled(enabled)
        self.simplify_btn.setEnabled(enabled)
        self.ocr_btn.setEnabled(enabled)
        self.georef_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
    
    def update_parameters(self):
        """Update module parameters from GUI"""
        self.preprocessor.threshold_value = self.threshold_spin.value()
        self.simplifier.epsilon = self.epsilon_spin.value()
        self.ocr_detector.confidence_threshold = self.confidence_spin.value()
    
    def log_message(self, msg):
        """Add message to status log"""
        self.status_text.append(msg)
        QApplication.processEvents()
    
    def clear_project(self):
        """Clear all data and reset the project"""
        reply = QMessageBox.question(
            self, 'Clear Project',
            'Are you sure you want to clear all data?\nThis will reset everything.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.log_message("=" * 50)
            self.log_message("Clearing project...")
            
            # Clear all data
            self.image = None
            self.processed_image = None
            self.skeleton = None
            self.contours = None
            self.simplified_contours = None
            self.elevations = None
            self.georeferenced = None
            self.current_file = None
            
            # Clear display
            self.ax.clear()
            self.ax.set_title("Ready")
            self.ax.axis('off')
            self.canvas.draw()
            
            # Clear status
            self.status_text.clear()
            
            # Reset modules
            self.reader = None
            self.preprocessor = Preprocessor()
            self.skeletonizer = Skeletonizer()
            self.graph_analyzer = GraphAnalyzer()
            self.tracker = ContourTracker()
            self.simplifier = PolylineSimplifier()
            self.georeferencer = Georeferencer()
            self.dxf_exporter = DXFExporter()
            self.ocr_detector = OCRDetector()
            
            # Reset parameters
            self.threshold_spin.setValue(128)
            self.epsilon_spin.setValue(1.0)
            self.confidence_spin.setValue(60)
            self.method_combo.setCurrentIndex(0)
            
            self.log_message("Project cleared successfully!")
            self.log_message("You can now load a new image.")
            self.log_message("=" * 50)
    
    def load_image(self):
        """Load TIFF image"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF File", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        
        if not filepath:
            return
            
        try:
            self.log_message(f"Loading: {filepath}")
            self.current_file = filepath
            
            self.reader = RasterReader(filepath, max_size=2048)
            self.image = self.reader.read()
            
            if self.image is None:
                self.log_message("Error: Failed to load image")
                return
            
            self.log_message(f"Image loaded: {self.image.shape}")
            self.log_message(f"Memory usage: {self.image.nbytes / 1024 / 1024:.2f} MB")
            
            if self.reader.transform:
                self.georeferencer.set_georeferencing(
                    self.reader.transform,
                    self.reader.pixel_size,
                    self.reader.crs
                )
                self.log_message("Georeferencing information found")
            
            self.display_image(self.image, "Original Image")
            
        except Exception as e:
            self.log_message(f"Error loading image: {str(e)}")
    
    def display_image(self, image, title="Image"):
        """Display image in matplotlib canvas"""
        self.ax.clear()
        
        if len(image.shape) == 3:
            self.ax.imshow(image)
        else:
            self.ax.imshow(image, cmap='gray')
            
        self.ax.set_title(title)
        self.ax.axis('off')
        self.canvas.draw()
    
    def preprocess_image(self):
        """Apply preprocessing"""
        if self.image is None:
            self.log_message("Error: No image loaded")
            return
            
        try:
            self.log_message("Preprocessing image...")
            self.processed_image = self.preprocessor.process(self.image)
            
            # Έλεγχος αν η εικόνα είναι μαύρη
            white_pixels = np.sum(self.processed_image > 0)
            if white_pixels == 0:
                self.log_message("⚠️ WARNING: Preprocessing resulted in completely black image!")
                self.log_message("⚠️ Try skipping preprocessing or adjust threshold.")
            else:
                self.log_message(f"Preprocessing complete: {white_pixels} white pixels")
            
            self.display_image(self.processed_image, "Preprocessed Image")
            
        except Exception as e:
            self.log_message(f"Error in preprocessing: {str(e)}")
    
    def skeletonize_image(self):
        """Apply skeletonization"""
        image_to_use = self.processed_image if self.processed_image is not None else self.image
        
        if image_to_use is None:
            self.log_message("Error: No image loaded")
            return
            
        try:
            self.log_message("Skeletonizing...")
            self.skeleton = self.skeletonizer.skeletonize(image_to_use)
            
            if self.skeleton is None or np.sum(self.skeleton > 0) == 0:
                self.log_message("⚠️ WARNING: Skeleton is empty!")
                self.log_message("⚠️ Try using 'Skip Skeleton' button instead.")
            else:
                self.log_message(f"Skeletonization complete: {np.sum(self.skeleton > 0)} pixels")
            
            self.display_image(self.skeleton, "Skeleton")
            
        except Exception as e:
            self.log_message(f"Error in skeletonization: {str(e)}")
    
    def skip_to_contours(self):
        """Skip skeletonization and go directly to contour extraction"""
        if self.image is None:
            self.log_message("Error: No image loaded")
            return
        
        self.log_message("Skipping skeletonization...")
        self.log_message("Using original/preprocessed image directly for contour extraction")
        
        # Χρησιμοποιούμε την preprocessed ή την αρχική εικόνα
        image_to_use = self.processed_image if self.processed_image is not None else self.image
        
        # Κάνουμε threshold αν χρειάζεται
        if image_to_use.max() > 1:
            _, image_to_use = cv2.threshold(image_to_use, 128, 255, cv2.THRESH_BINARY)
        
        # Αποθηκεύουμε ως skeleton (για να το χρησιμοποιήσει το extract_contours)
        self.skeleton = image_to_use
        
        self.log_message(f"Ready for contour extraction: {np.sum(self.skeleton > 0)} white pixels")
        self.display_image(self.skeleton, "Ready for Contour Extraction")
        
        # Αυτόματα πάμε στο extract contours
        self.extract_contours()
    
    def extract_contours(self):
        """Extract contours from skeleton or image"""
        # Αν δεν υπάρχει skeleton, χρησιμοποιούμε την εικόνα
        if self.skeleton is None:
            image_to_use = self.processed_image if self.processed_image is not None else self.image
            
            if image_to_use is None:
                self.log_message("Error: No image loaded")
                return
            
            self.log_message("No skeleton found, using image directly...")
            if image_to_use.max() > 1:
                _, image_to_use = cv2.threshold(image_to_use, 128, 255, cv2.THRESH_BINARY)
            self.skeleton = image_to_use
        
        if self.skeleton is None:
            self.log_message("Error: No image or skeleton available")
            return
            
        try:
            self.log_message("Extracting contours...")
            
            method = self.method_combo.currentText()
            use_cv2 = method == "OpenCV"
            
            if method == "Auto":
                total_pixels = np.sum(self.skeleton > 0)
                use_cv2 = total_pixels < 5000
                self.log_message(f"Auto: {'OpenCV' if use_cv2 else 'Graph'}")
            
            self.contours = self.tracker.track_contours(self.skeleton, use_cv2=use_cv2)
            self.contours = self.tracker.merge_contours(self.contours, max_gap=10)
            
            if len(self.contours) == 0:
                self.log_message("⚠️ WARNING: No contours found!")
                self.log_message("⚠️ Try using 'Skip Skeleton' button or different extraction method")
            else:
                self.log_message(f"Extracted {len(self.contours)} contours")
            
            self.display_contours(self.contours)
            
        except Exception as e:
            self.log_message(f"Error in contour extraction: {str(e)}")
            # Fallback
            try:
                self.log_message("Trying OpenCV fallback...")
                self.contours = self.tracker.track_contours(self.skeleton, use_cv2=True)
                self.log_message(f"Extracted {len(self.contours)} contours with OpenCV")
                self.display_contours(self.contours)
            except Exception as e2:
                self.log_message(f"Fallback failed: {str(e2)}")
    
    def display_contours(self, contours):
        """Display extracted contours on image"""
        self.ax.clear()
        
        # Show original or preprocessed image
        if self.processed_image is not None:
            self.ax.imshow(self.processed_image, cmap='gray')
        elif self.image is not None:
            self.ax.imshow(self.image, cmap='gray')
        
        # Draw contours
        if contours:
            for contour in contours:
                if len(contour) > 1:
                    points = np.array([(x, y) for y, x in contour])
                    self.ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=0.8)
            self.ax.set_title(f"Contours ({len(contours)})")
        else:
            self.ax.set_title("No Contours Found")
        
        self.ax.axis('off')
        self.canvas.draw()
    
    def simplify_polylines(self):
        """Simplify polylines"""
        if self.contours is None or len(self.contours) == 0:
            self.log_message("Error: No contours found. Extract Contours first.")
            return
            
        try:
            self.log_message("Simplifying polylines...")
            
            self.simplified_contours = []
            for contour in self.contours:
                simplified = self.simplifier.simplify(
                    contour, 
                    method='douglas_peucker',
                    epsilon=self.epsilon_spin.value()
                )
                self.simplified_contours.append(simplified)
            
            self.log_message(f"Simplified to {len(self.simplified_contours)} contours")
            self.display_contours(self.simplified_contours)
            
        except Exception as e:
            self.log_message(f"Error in simplification: {str(e)}")
    
    def detect_elevations(self):
        """Detect elevations using OCR"""
        if self.image is None:
            self.log_message("Error: No image loaded")
            return
            
        try:
            self.log_message("Detecting elevations with OCR...")
            
            self.elevations = self.ocr_detector.detect_elevations(self.image)
            self.log_message(f"Detected {len(self.elevations)} elevation values")
            
            if self.elevations:
                values = [e['value'] for e in self.elevations[:10]]
                self.log_message(f"Elevations found: {values}")
                
                contours_to_use = self.simplified_contours if self.simplified_contours else self.contours
                if contours_to_use:
                    assignments = self.ocr_detector.map_elevations_to_contours(
                        contours_to_use, self.elevations
                    )
                    assigned = sum(1 for a in assignments if a != 0)
                    self.log_message(f"Assigned elevations to {assigned} contours")
            
        except Exception as e:
            self.log_message(f"Error in OCR: {str(e)}")
            self.log_message("Make sure Tesseract is installed")
    
    def georeference_contours(self):
        """Georeference contours"""
        contours_to_use = self.simplified_contours if self.simplified_contours else self.contours
        
        if contours_to_use is None or len(contours_to_use) == 0:
            self.log_message("Error: No contours to georeference")
            return
            
        try:
            self.log_message("Georeferencing...")
            
            if self.georeferencer.transform is None:
                self.log_message("⚠️ No georeferencing information found")
                self.log_message("   Using pixel coordinates (without georeferencing)")
                self.georeferenced = None  # Δεν κάνουμε georeferencing
                self.log_message("   You can still export DXF with pixel coordinates")
                return
            else:
                # Georeferencing με αναστροφή Y
                image_height = self.image.shape[0] if self.image is not None else None
                self.georeferenced = self.georeferencer.transform_contours(
                    contours_to_use, 
                    flip_y=True, 
                    image_height=image_height
                )
                self.log_message(f"✅ Georeferenced {len(self.georeferenced)} contours to EGSA87")
                
                if self.georeferenced and self.georeferenced[0]:
                    first = self.georeferenced[0][0]
                    self.log_message(f"   First point: X={first[0]:.2f}, Y={first[1]:.2f}")
                
        except Exception as e:
            self.log_message(f"Error in georeferencing: {str(e)}")
            self.log_message("   Using pixel coordinates as fallback")
            self.georeferenced = None
    
    def export_dxf(self):
        """Export to DXF with proper coordinate handling"""
        # Επιλογή των σωστών contours
        if self.georeferenced is not None and len(self.georeferenced) > 0:
            contours_to_export = self.georeferenced
            has_georeferencing = True
            self.log_message("Exporting georeferenced contours (EGSA87)")
        else:
            contours_to_export = self.simplified_contours if self.simplified_contours else self.contours
            has_georeferencing = False
            self.log_message("Exporting pixel coordinates (no georeferencing)")
    
        if contours_to_export is None or len(contours_to_export) == 0:
            self.log_message("Error: No contours to export")
            return
    
        # Ερώτηση για το αν θέλουμε να χρησιμοποιήσουμε 3D
        use_3d = False
        if self.elevations and len(self.elevations) > 0:
            reply = QMessageBox.question(
                self, '3D Export',
                'Elevation data found. Export as 3D DXF?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            use_3d = (reply == QMessageBox.Yes)
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export DXF", "", "DXF Files (*.dxf)"
        )
        
        if not filepath:
            return
        
        try:
            self.log_message(f"Exporting to {filepath}...")
            
            # Προετοιμασία παραμέτρων
            image_height = self.image.shape[0] if self.image is not None else None
            
            self.dxf_exporter.image_height = image_height
            self.dxf_exporter.use_3d = use_3d
            
            # Elevations
            elevations = [0] * len(contours_to_export)
            if self.elevations and use_3d:
                # Προσπάθεια αντιστοίχισης elevations
                try:
                    elevations = self.ocr_detector.map_elevations_to_contours(
                        contours_to_export, self.elevations
                    )
                    self.log_message(f"Using {sum(1 for e in elevations if e != 0)} elevations")
                except:
                    self.log_message("Warning: Could not map elevations, using 0")
            
            # Εξαγωγή
            exported = self.dxf_exporter.export_contours(
                contours_to_export,
                elevations,
                filepath,
                image_height=image_height,
                has_georeferencing=has_georeferencing
            )
            
            self.log_message(f"✅ Export complete: {filepath}")
            self.log_message(f"✅ Exported {exported} contours")
        
            if has_georeferencing:
                self.log_message("✅ Coordinates: EGSA87 (georeferenced)")
            else:
                self.log_message("✅ Coordinates: Pixel coordinates (Y-axis corrected)")
                
        except Exception as e:
            self.log_message(f"Error exporting: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
    
    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(
            self, 'Exit',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    window = TopoVectorizerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
