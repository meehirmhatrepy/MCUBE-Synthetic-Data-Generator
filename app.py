import sys
import os
import math
import random
import json
from functools import partial

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QHBoxLayout,
    QVBoxLayout, QWidget, QSlider, QSpinBox, QMessageBox, QLineEdit, QGroupBox,
    QFormLayout, QColorDialog, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont
from extract_instance_rgba import extract_instance_rgba
from poly_to_mask import poly_to_mask
from transform_polygon import transform_polygon
from polygon_area import polygon_area
from read_yolo_seg_txt import read_yolo_seg_txt
from read_coco_json import read_coco_json
from write_coco_json import write_coco_json
from paste_transformed_rgba_roi import paste_transformed_rgba_roi
# ------------------ GUI ------------------

class SyntheticGenerator(QMainWindow):
    def __init__(self):
        """
        Initialize the main window, theme, and all state variables.
        """
        super().__init__()
        self.setWindowTitle('MCUBE Synthetic Data Generator')
        self.setGeometry(80, 80, 1280, 840)
    
        # --- Modern Theme ---
        self.setStyleSheet("""
            QMainWindow {
                background: #121629;
                color: #f6f6f6;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 16px;
            }
            QWidget {
                background-color: transparent;
            }

            /* General labels (e.g., Brightness, Sharpness) */
            QLabel {
                color: #ffffff;
                font-weight: bold;
                padding: 2px;
                /* Subtle text shadow for better visibility */
                
            }

            /* Header bar */
            QLabel#Header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #762dfd, stop:1 #1f9ecc);
                font-size: 28px;
                font-weight: 700;
                color: #fff;
                letter-spacing: 2px;
                padding: 14px;
                border-radius: 12px;
                margin-bottom: 12px;
            }

            /* Image display area */
            QLabel#ImageArea {
                background: #1c2237;
                color: #eee;
                border-radius: 16px;
                border: 2px solid #1f9ecc;
            }

            /* Group boxes */
            QGroupBox {
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 12px;
                margin-top: 12px;
                background: rgba(255,255,255,0.05);
                padding-top: 16px;
                font-size: 17px;
                font-weight: 500;
                color: #fff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 4px 8px;
                background-color: rgba(31,158,204,0.3);
                border-radius: 8px;
                color: #fff;
            }

            /* Buttons */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #762dfd, stop:1 #1f9ecc);
                color: #fff;
                border: none;
                border-radius: 14px;
                padding: 8px 20px;
                font-size: 15px;
                font-weight: 600;
                margin: 4px 0;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1f9ecc, stop:1 #762dfd);
            }

            /* Inputs */
            QLineEdit, QComboBox, QSpinBox {
                background: #232a3d;
                color: #fff;
                border-radius: 8px;
                border: 1px solid #1f9ecc;
                padding: 4px 8px;
            }

            /* Sliders */
            QSlider::groove:horizontal {
                height: 8px;
                background: #1f9ecc;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #762dfd;
                border: 2px solid #fff;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
        """)



        # --- Image enhancement params ---
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.sharpness = 1.0

        self.img = None
        self.instances = []  # each inst: {'class','orig_poly','poly','angle','scale','translate'}
        self.selected_idx = None
        self.dragging = False
        self.last_mouse = None
        self.scale_factor = 1.0

        # clipboard for copy/cut/paste
        self.clipboard = None  # {'rgba','origin','poly','class'}
        self.clipboard_mode = False
        self.clipboard_pos = None  # (x,y) in image coords for preview

        # cut fill color
        self.cut_fill = (0,0,0)  # default black

        self._build_ui()

    def _build_ui(self):
        """
        Build and layout all UI widgets and controls.
        """
        main = QWidget()
        self.setCentralWidget(main)
        hb = QHBoxLayout()
        main.setLayout(hb)

        # --- Image Display ---
        self.img_label = QLabel('Load image and label to start')
        self.img_label.setMinimumSize(800, 800)    
        self.img_label.setStyleSheet('background: rgba(35,42,61,0.95); color: #eee; border-radius: 24px; border: 2px solid #1f9ecc;')
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMouseTracking(True)
        self.img_label.mousePressEvent = self.img_mouse_press
        self.img_label.mouseMoveEvent = self.img_mouse_move
        self.img_label.mouseReleaseEvent = self.img_mouse_release

        # --- Controls ---
        ctrl = QVBoxLayout()
        ctrl.setSpacing(18)

        header = QLabel("MCUBE Synthetic Data Generator")
        header.setObjectName("Header")
        header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #762dfd, stop:1 #1f9ecc);
            color: #fff;
            border-radius: 16px;
            padding: 18px 0 18px 0;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 2px;
        """)
        ctrl.addWidget(header)

        # --- File Buttons ---
        btn_load_img = QPushButton('Upload Image')
        btn_load_img.clicked.connect(self.upload_image)
        ctrl.addWidget(btn_load_img)

        btn_load_txt = QPushButton('Upload YOLO .txt')
        btn_load_txt.clicked.connect(self.upload_txt)
        ctrl.addWidget(btn_load_txt)

        btn_load_coco = QPushButton('Upload COCO .json')
        btn_load_coco.clicked.connect(self.upload_coco_json)
        ctrl.addWidget(btn_load_coco)

        # --- Enhancement controls ---
        enh_grp = QGroupBox('Image Enhancement')
        enh_form = QFormLayout()
        enh_grp.setLayout(enh_form)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setMinimum(-100); self.slider_brightness.setMaximum(100); self.slider_brightness.setValue(0)
        self.slider_brightness.valueChanged.connect(self.on_enhance_change)
        enh_form.addRow('Brightness:', self.slider_brightness)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setMinimum(10); self.slider_contrast.setMaximum(300); self.slider_contrast.setValue(100)
        self.slider_contrast.valueChanged.connect(self.on_enhance_change)
        enh_form.addRow('Contrast:', self.slider_contrast)

        self.slider_saturation = QSlider(Qt.Horizontal)
        self.slider_saturation.setMinimum(10); self.slider_saturation.setMaximum(300); self.slider_saturation.setValue(100)
        self.slider_saturation.valueChanged.connect(self.on_enhance_change)
        enh_form.addRow('Saturation:', self.slider_saturation)

        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setMinimum(10); self.slider_sharpness.setMaximum(300); self.slider_sharpness.setValue(100)
        self.slider_sharpness.valueChanged.connect(self.on_enhance_change)
        enh_form.addRow('Sharpness:', self.slider_sharpness)

        ctrl.addWidget(enh_grp)

        # --- Copy/Cut/Paste ---
        hcopy = QHBoxLayout()
        btn_copy = QPushButton('Copy')
        btn_copy.clicked.connect(self.copy_selected)
        btn_cut = QPushButton('Cut')
        btn_cut.clicked.connect(self.cut_selected)
        btn_paste = QPushButton('Paste')
        btn_paste.clicked.connect(self.enter_paste_mode)
        hcopy.addWidget(btn_copy); hcopy.addWidget(btn_cut); hcopy.addWidget(btn_paste)
        ctrl.addLayout(hcopy)

        # --- Cut Fill Color ---
        color_box = QHBoxLayout()
        btn_color = QPushButton('Cut Fill Color')
        btn_color.clicked.connect(self.pick_cut_color)
        self.fill_choice = QComboBox()
        self.fill_choice.addItems(['Black','Mean Color'])
        color_box.addWidget(btn_color); color_box.addWidget(self.fill_choice)
        ctrl.addLayout(color_box)

        # --- Transform Controls ---
        grp = QGroupBox('Transform Selected')
        form = QFormLayout()
        grp.setLayout(form)

        self.slider_rotate = QSlider(Qt.Horizontal)
        self.slider_rotate.setMinimum(-180); self.slider_rotate.setMaximum(180); self.slider_rotate.setValue(0)
        self.slider_rotate.valueChanged.connect(self.on_transform_change)
        form.addRow('Rotate (deg):', self.slider_rotate)

        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setMinimum(10); self.slider_scale.setMaximum(300); self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.on_transform_change)
        form.addRow('Scale (%):', self.slider_scale)

        btn_reset = QPushButton('Reset Transform')
        btn_reset.clicked.connect(self.reset_transform)
        form.addRow(btn_reset)

        ctrl.addWidget(grp)

        # --- Save/Export ---
        btn_save_coco = QPushButton('Save Current (COCO .json)')
        btn_save_coco.clicked.connect(self.save_current_coco)
        ctrl.addWidget(btn_save_coco)

        # --- Random Variations ---
        self.class_selector = QComboBox()
        self.class_selector.addItem("Select Class")
        self.class_selector.setMinimumWidth(120)
        ctrl.addWidget(self.class_selector)

        hrand = QHBoxLayout()
        self.spin_n = QSpinBox(); self.spin_n.setMinimum(1); self.spin_n.setMaximum(1000); self.spin_n.setValue(10)
        btn_rand = QPushButton('Generate Random Variations')
        btn_rand.clicked.connect(self.generate_random_variations)
        hrand.addWidget(self.spin_n); hrand.addWidget(btn_rand)
        ctrl.addLayout(hrand)

        # --- Export Directory ---
        exp_layout = QHBoxLayout()
        default_export_dir = os.path.join(os.getcwd(), "generated images and annos")
        if not os.path.exists(default_export_dir):
            os.makedirs(default_export_dir)
        self.line_export = QLineEdit(default_export_dir)
        btn_browse = QPushButton('Browse')
        btn_browse.clicked.connect(self.browse_export)
        exp_layout.addWidget(self.line_export); exp_layout.addWidget(btn_browse)
        ctrl.addLayout(exp_layout)

        ctrl.addStretch()

      
        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setMinimumWidth(320)  

        hb.addWidget(self.img_label, stretch=2)   
        hb.addWidget(ctrl_widget, stretch=1)      

    # ----------------- I/O -----------------
    def upload_image(self):
        """
        Open a file dialog to select and load an image.
        Resets annotation and clipboard state.
        """
        path, _ = QFileDialog.getOpenFileName(self, 'Select image', '', 'Images (*.png *.jpg *.jpeg)')
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, 'Error', 'Failed to read image')
            return
        self.image_path = path
        self.img = img
        self.img_h, self.img_w = img.shape[:2]
        self.instances = []
        self.selected_idx = None
        self.clipboard = None
        self.clipboard_mode = False
        self.draw_image()

    def upload_txt(self):
        """
        Open a file dialog to select and load YOLO segmentation .txt file.
        Updates instance list and class selector.
        """
        path, _ = QFileDialog.getOpenFileName(self, 'Select YOLO .txt', '', 'Text files (*.txt)')
        if not path:
            return
        if self.img is None:
            QMessageBox.warning(self, 'Warning', 'Load image first')
            return
        insts = read_yolo_seg_txt(path, self.img_w, self.img_h)
        self.instances = []
        class_set = set()
        for inst in insts:
            poly = inst['poly']
            self.instances.append({
                'class': inst['class'],
                'orig_poly': poly.copy(),
                'poly': poly.copy(),
                'angle': 0.0,
                'scale': 1.0,
                'translate': np.array([0.0, 0.0])
            })
            class_set.add(inst['class'])
        # Update class_selector dropdown
        self.class_selector.clear()
        self.class_selector.addItem("Select Class")
        for c in sorted(class_set):
            self.class_selector.addItem(str(c))
        self.selected_idx = 0 if len(self.instances)>0 else None
        self.reset_transform()
        self.draw_image()

    def upload_coco_json(self):
        """
        Open a file dialog to select and load COCO .json annotation file.
        Updates instance list and class selector.
        """
        path, _ = QFileDialog.getOpenFileName(self, 'Select COCO .json', '', 'JSON files (*.json)')
        if not path:
            return
        if self.img is None:
            QMessageBox.warning(self, 'Warning', 'Load image first')
            return
        try:
            insts = read_coco_json(path, os.path.basename(self.image_path), self.img_w, self.img_h)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to read COCO JSON: {e}')
            return
        self.instances = []
        class_set = set()
        for inst in insts:
            poly = inst['poly']
            cls = inst.get('class', 0)
            self.instances.append({
                'class': cls,
                'orig_poly': poly.copy(),
                'poly': poly.copy(),
                'angle': 0.0,
                'scale': 1.0,
                'translate': np.array([0.0, 0.0])
            })
            class_set.add(cls)
        # Update class_selector dropdown
        self.class_selector.clear()
        self.class_selector.addItem("Select Class")
        for c in sorted(class_set):
            self.class_selector.addItem(str(c))
        self.selected_idx = 0 if len(self.instances)>0 else None
        self.reset_transform()
        self.draw_image()

    def browse_export(self):
        """
        Open a folder dialog to select export directory.
        """
        path = QFileDialog.getExistingDirectory(self, 'Select Export Folder', os.getcwd())
        if path:
            self.line_export.setText(path)

    def pick_cut_color(self):
        """
        Open a color picker dialog to set the fill color for cut operations.
        """
        col = QColorDialog.getColor()
        if col.isValid():
            self.cut_fill = (col.blue(), col.green(), col.red())

    # ----------------- Clipboard -----------------
    def copy_selected(self):
        """
        Copy the currently selected instance to the clipboard.
        """
        if self.selected_idx is None:
            QMessageBox.warning(self, 'Info', 'Select an instance to copy')
            return
        inst = self.instances[self.selected_idx]
        poly = inst['orig_poly']
        xs = poly[:,0]; ys = poly[:,1]
        minx = int(xs.min()); miny = int(ys.min())
        rgba, origin = extract_instance_rgba(self.img, poly)
        if rgba is None:
            QMessageBox.warning(self, 'Info', 'Failed to extract instance')
            return
        self.clipboard = {
            'rgba': rgba.copy(),
            'origin': origin,
            'poly': poly.copy(),
            'class': inst['class'],
            'bbox_topleft': (minx, miny)
        }
        QMessageBox.information(self, 'Copied', 'Instance copied to clipboard')

    def cut_selected(self):
        """
        Cut the currently selected instance to the clipboard and fill the region.
        """
        if self.selected_idx is None:
            QMessageBox.warning(self, 'Info', 'Select an instance to cut')
            return
        inst = self.instances[self.selected_idx]
        rgba, origin = extract_instance_rgba(self.img, inst['orig_poly'])
        if rgba is None:
            QMessageBox.warning(self, 'Info', 'Failed to extract instance')
            return
        self.clipboard = {'rgba': rgba.copy(), 'origin': origin, 'poly': inst['orig_poly'].copy(), 'class': inst['class']}
        # blank area in original image
        mask = poly_to_mask(inst['orig_poly'], self.img_h, self.img_w)
        if self.fill_choice.currentText() == 'Black':
            self.img[mask>0] = (0,0,0)
        else:
            # mean color in bounding box
            xs = inst['orig_poly'][:,0]; ys = inst['orig_poly'][:,1]
            minx = max(0, int(xs.min())); miny = max(0, int(ys.min()))
            maxx = min(self.img_w, int(xs.max())+1); maxy = min(self.img_h, int(ys.max())+1)
            roi = self.img[miny:maxy, minx:maxx]
            if roi.size == 0:
                fill = (0,0,0)
            else:
                mean = cv2.mean(roi)[:3]
                fill = (int(mean[0]), int(mean[1]), int(mean[2]))
            self.img[mask>0] = fill
        # remove instance
        del self.instances[self.selected_idx]
        self.selected_idx = None
        QMessageBox.information(self, 'Cut', 'Instance cut to clipboard')
        self.draw_image()

    def enter_paste_mode(self):
        """
        Enable clipboard paste preview mode.
        """
        if self.clipboard is None:
            QMessageBox.warning(self, 'Info', 'Clipboard empty. Copy or cut first')
            return
        self.clipboard_mode = True
        # initial preview position set to center
        self.clipboard_pos = (self.img_w//2, self.img_h//2)
        self.draw_image()

    def place_clipboard_at(self, x, y):
        """
        Paste the clipboard instance at the given (x, y) location.
        """
        poly = self.clipboard['poly'].copy()
        minx, miny = self.clipboard['bbox_topleft']
        # Compute translation so that bbox top-left moves to (x, y)
        translate = np.array([x, y]) - np.array([minx, miny])
        newpoly = poly + translate
        self.instances.append({
            'class': self.clipboard['class'],
            'orig_poly': poly.copy(),
            'poly': newpoly,
            'angle': 0.0,
            'scale': 1.0,
            'translate': translate
        })
        self.clipboard_mode = False
        self.selected_idx = len(self.instances)-1
        self.draw_image()
  
    # ----------------- Interaction -----------------
    def img_mouse_press(self, ev):
        """
        Handle mouse press events on the image label for selection and paste.
        """
        if self.img is None:
            return
        pos = ev.pos()
        ix, iy = self.display_to_image(pos.x(), pos.y())
        if ix is None or ix < 0 or iy is None or iy < 0 or ix >= self.img_w or iy >= self.img_h:
            return
        if self.clipboard_mode:
            # place clipboard
            self.place_clipboard_at(ix, iy)
            return
        # selection logic: choose smallest area polygon containing click
        candidates = []
        for i, inst in enumerate(self.instances):
            mask = poly_to_mask(inst['poly'], self.img_h, self.img_w)
            if mask[int(iy), int(ix)] > 0:
                candidates.append((i, polygon_area(inst['poly'])))
        if not candidates:
            self.selected_idx = None
            self.draw_image()
            return
        candidates.sort(key=lambda x: x[1])
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & Qt.ControlModifier) and hasattr(self, 'last_candidates') and self.last_candidates:
            prev_idxs = [c[0] for c in self.last_candidates]
            if self.selected_idx in prev_idxs:
                cur_pos = prev_idxs.index(self.selected_idx)
                next_idx = prev_idxs[(cur_pos + 1) % len(prev_idxs)]
                self.selected_idx = next_idx
            else:
                self.selected_idx = candidates[0][0]
        else:
            self.selected_idx = candidates[0][0]
        self.last_candidates = candidates
        self.dragging = True
        self.last_mouse = (ix, iy)
        self.draw_image()

    def img_mouse_move(self, ev):
        """
        Handle mouse move events for dragging and clipboard preview.
        """
        pos = ev.pos()
        ix, iy = self.display_to_image(pos.x(), pos.y())
        if self.clipboard_mode:
            # update preview pos
            if ix is not None and iy is not None and ix>=0 and iy>=0 and ix<self.img_w and iy<self.img_h:
                self.clipboard_pos = (ix, iy)
                self.draw_image()
            return
        if not self.dragging or self.selected_idx is None:
            return
        if ix is None:
            return
        dx = ix - self.last_mouse[0]
        dy = iy - self.last_mouse[1]
        self.last_mouse = (ix, iy)
        inst = self.instances[self.selected_idx]
        inst['translate'] = inst['translate'] + np.array([dx, dy])
        xs = inst['orig_poly'][:,0]; ys = inst['orig_poly'][:,1]
        minx = xs.min(); maxx = xs.max()
        miny = ys.min(); maxy = ys.max()
        bbox_center = np.array([(minx + maxx) / 2, (miny + maxy) / 2], dtype=np.float32)
        inst['poly'] = transform_polygon(
            inst['orig_poly'],
            translate=tuple(inst['translate']),
            angle_deg=inst['angle'],
            scale=inst['scale'],
            origin=bbox_center
        )
        self.draw_image()

    def img_mouse_release(self, ev):
        """
        Handle mouse release events to stop dragging.
        """
        self.dragging = False
        self.last_mouse = None

    def display_to_image(self, dx, dy):
        """
        Convert display coordinates to image coordinates.
        """
        lbl_w = self.img_label.width(); lbl_h = self.img_label.height()
        if self.img is None:
            return None, None
        img_h, img_w = self.img_h, self.img_w
        scale = min(lbl_w / img_w, lbl_h / img_h)
        self.scale_factor = scale
        new_w = int(img_w * scale); new_h = int(img_h * scale)
        x0 = (lbl_w - new_w)//2; y0 = (lbl_h - new_h)//2
        if dx < x0 or dy < y0 or dx >= x0 + new_w or dy >= y0 + new_h:
            return -1, -1
        ix = (dx - x0) / scale
        iy = (dy - y0) / scale
        return int(ix), int(iy)

    # ----------------- Transform / Draw -----------------
    def on_transform_change(self):
        """
        Update the selected instance's rotation and scale from sliders.
        """
        if self.selected_idx is None:
            return
        angle = self.slider_rotate.value()
        scale_pc = self.slider_scale.value()
        scale = scale_pc / 100.0
        inst = self.instances[self.selected_idx]
        inst['angle'] = float(angle)
        inst['scale'] = float(scale)
        xs = inst['orig_poly'][:,0]; ys = inst['orig_poly'][:,1]
        minx = xs.min(); maxx = xs.max()
        miny = ys.min(); maxy = ys.max()
        bbox_center = np.array([(minx + maxx) / 2, (miny + maxy) / 2], dtype=np.float32)
        inst['poly'] = transform_polygon(
            inst['orig_poly'],
            translate=tuple(inst['translate']),
            angle_deg=inst['angle'],
            scale=inst['scale'],
            origin=bbox_center
        )
        self.draw_image()

    def reset_transform(self):
        """
        Reset the selected instance's transform to default.
        """
        if self.selected_idx is None:
            return
        inst = self.instances[self.selected_idx]
        inst['poly'] = inst['orig_poly'].copy()
        inst['angle'] = 0.0; inst['scale'] = 1.0; inst['translate'] = np.array([0.0,0.0])
        self.slider_rotate.setValue(0); self.slider_scale.setValue(100)
        self.draw_image()

    def enhance_image(self, img):
        """
        Apply brightness, contrast, saturation, and sharpness enhancements to the image.
        """
        # Brightness & Contrast
        img = cv2.convertScaleAbs(img, alpha=self.contrast, beta=self.brightness)
        # Saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,1] *= self.saturation
        hsv[...,1] = np.clip(hsv[...,1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Sharpness
        if self.sharpness != 1.0:
            kernel = np.array([[0, -1, 0], [-1, 5*self.sharpness, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)
        return img

    def on_enhance_change(self):
        """
        Update enhancement parameters from sliders and redraw image.
        """
        self.brightness = self.slider_brightness.value()
        self.contrast = self.slider_contrast.value() / 100.0
        self.saturation = self.slider_saturation.value() / 100.0
        self.sharpness = self.slider_sharpness.value() / 100.0
        self.draw_image()

    def draw_image(self):
        """
        Draw the enhanced image, all instances, clipboard preview, and outlines.
        """
        if self.img is None:
            self.img_label.setText('Load image and label to start')
            return

        # --- Apply enhancements ---
        canvas = self.enhance_image(self.img.copy())

        
        for i, inst in enumerate(self.instances):
            rgba, origin_src = extract_instance_rgba(canvas, inst['orig_poly'])
    # ...
            if rgba is None:
                continue

            # Use bounding box top-left for both polygon and ROI
            xs_src = inst['orig_poly'][:,0]; ys_src = inst['orig_poly'][:,1]
            minx_src = xs_src.min(); maxx_src = xs_src.max()
            miny_src = ys_src.min(); maxy_src = ys_src.max()
            bbox_center_src = np.array([(minx_src + maxx_src) / 2, (miny_src + maxy_src) / 2], dtype=np.float32)
            xs_dst = inst['poly'][:,0]; ys_dst = inst['poly'][:,1]
            minx_dst = xs_dst.min(); maxx_dst = xs_dst.max()
            miny_dst = ys_dst.min(); maxy_dst = ys_dst.max()
            bbox_center_dst = np.array([(minx_dst + maxx_dst) / 2, (miny_dst + maxy_dst) / 2], dtype=np.float32)
            angle = inst['angle']
            scale = inst['scale']

            canvas = paste_transformed_rgba_roi(
                canvas,
                rgba,
                origin_src,
                bbox_center_src,
                bbox_center_dst,
                angle,
                scale
            )

        # --- Clipboard preview ---
        if self.clipboard_mode and self.clipboard is not None and self.clipboard_pos is not None:
            rgba, origin_src = extract_instance_rgba(canvas, self.clipboard['poly'])
            origin_src = self.clipboard['origin']
            poly = self.clipboard['poly']
            xs = poly[:,0]; ys = poly[:,1]
            minx = xs.min(); maxx = xs.max()
            miny = ys.min(); maxy = ys.max()
            bbox_center_src = np.array([(minx + maxx) / 2, (miny + maxy) / 2], dtype=np.float32)
            bbox_center_dst = np.array(self.clipboard_pos, dtype=np.float32)

            preview = paste_transformed_rgba_roi(
                canvas.copy(),
                rgba,
                origin_src,
                bbox_center_src,
                bbox_center_dst,
                0.0,
                1.0
            )

            alpha_preview = 0.5
            canvas = cv2.addWeighted(preview, alpha_preview, canvas, 1 - alpha_preview, 0)

        # --- Draw polygon outlines ---
        disp = canvas.copy()
        for i, inst in enumerate(self.instances):
            pts = inst['poly'].astype(np.int32)
            color = (0, 255, 0) if i == self.selected_idx else (255, 0, 0)
            cv2.polylines(disp, [pts], isClosed=True, color=color, thickness=2)

        h, w = disp.shape[:2]
        qimg = QImage(disp.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pix)

    def save_current_coco(self):
        """
        Save the current image and annotations in COCO format to the export directory.
        """
        if self.img is None:
            return
        folder = self.line_export.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.critical(self, 'Error', 'Export folder invalid')
            return
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        out_img_path = os.path.join(folder, base + '_aug.png')
        out_json_path = os.path.join(folder, base + '_aug.json')
        final = self.img.copy()
        for inst in self.instances:
            rgba, origin = extract_instance_rgba(final, inst['orig_poly'])
            if rgba is None:
                continue
            xs_src = inst['orig_poly'][:,0]; ys_src = inst['orig_poly'][:,1]
            minx_src = xs_src.min(); maxx_src = xs_src.max()
            miny_src = ys_src.min(); maxy_src = ys_src.max()
            bbox_center_src = np.array([(minx_src + maxx_src) / 2, (miny_src + maxy_src) / 2], dtype=np.float32)
            xs_dst = inst['poly'][:,0]; ys_dst = inst['poly'][:,1]
            minx_dst = xs_dst.min(); maxx_dst = xs_dst.max()
            miny_dst = ys_dst.min(); maxy_dst = ys_dst.max()
            bbox_center_dst = np.array([(minx_dst + maxx_dst) / 2, (miny_dst + maxy_dst) / 2], dtype=np.float32)
            final = paste_transformed_rgba_roi(final, rgba, origin, bbox_center_src, bbox_center_dst, inst['angle'], inst['scale'])
        cv2.imwrite(out_img_path, final)
        print(f"Saved image: {out_img_path}")
        write_coco_json(out_json_path, out_img_path, self.img_w, self.img_h, [{'class':inst['class'],'poly':inst['poly']} for inst in self.instances])
        QMessageBox.information(self, 'Saved', f'Saved image: {out_img_path}Saved COCO: {out_json_path}')

    def generate_random_variations(self):
        """
        Generate random variations of selected class instances and save images/annotations.
        """
        if self.img is None:
            QMessageBox.warning(self, 'Warning', 'Load image first')
            return
        selected_class = self.class_selector.currentText()
        if selected_class == "Select Class":
            QMessageBox.warning(self, 'Warning', 'Select a semantic class')
            return
        selected_class = int(selected_class)
        n = int(self.spin_n.value())
        folder = self.line_export.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.critical(self, 'Error', 'Export folder invalid')
            return
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        # Find all instances of selected class
        target_idxs = [i for i, inst in enumerate(self.instances) if inst['class'] == selected_class]
        if not target_idxs:
            QMessageBox.warning(self, 'Warning', f'No instances of class {selected_class} found')
            return
        for i in range(n):
            # Start with original image and original instances
            final = self.img.copy()
            new_instances = [dict(inst) for inst in self.instances]  # shallow copy
            for idx in target_idxs:
                inst = self.instances[idx]
                orig_poly = inst['orig_poly']
                xs = orig_poly[:,0]; ys = orig_poly[:,1]
                minx = int(xs.min()); miny = int(ys.min())
                # Random position within image bounds
                w = int(xs.max()) - minx
                h = int(ys.max()) - miny
                tx = random.uniform(0, max(0, self.img_w - w))
                ty = random.uniform(0, max(0, self.img_h - h))
                # No rotation/scale for pixel-perfect paste (add if needed)
                translate = np.array([tx, ty]) - np.array([minx, miny])
                newpoly = orig_poly + translate
                # Extract RGBA from original instance
                rgba, origin = extract_instance_rgba(self.img, orig_poly)
                if rgba is None:
                    continue
                # Paste using bbox top-left as reference
                origin_src = np.array([minx, miny])
                origin_dst = np.array([int(tx), int(ty)])
                final = paste_transformed_rgba_roi(final, rgba, origin_src, origin_src, origin_dst, 0.0, 1.0)
                # Add new instance annotation
                new_instances.append({
                    'class': inst['class'],
                    'orig_poly': orig_poly.copy(),
                    'poly': newpoly,
                    'angle': 0.0,
                    'scale': 1.0,
                    'translate': translate
                })
            # Save image and annotation
            out_img_path = os.path.join(folder, f"{base}_aug_{i:03d}.png")
            out_json_path = os.path.join(folder, f"{base}_aug_{i:03d}.json")
            cv2.imwrite(out_img_path, final)
            write_coco_json(out_json_path, out_img_path, self.img_w, self.img_h, [{'class':inst['class'],'poly':inst['poly']} for inst in new_instances])
        QMessageBox.information(self, 'Done', f'Generated {n} images in {folder}')


# Keyboard shortcuts
    def keyPressEvent(self, e):
        """
        Handle keyboard shortcuts for copy, cut, and paste.
        """
        if e.modifiers() == Qt.ControlModifier and e.key() == Qt.Key_C:
            self.copy_selected()
        if e.modifiers() == Qt.ControlModifier and e.key() == Qt.Key_X:
            self.cut_selected()
        if e.modifiers() == Qt.ControlModifier and e.key() == Qt.Key_V:
            self.enter_paste_mode()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SyntheticGenerator()
    win.show()
    print("Starting event loop")
    exit_code = app.exec_()
    print(f"Exited with code {exit_code}")

