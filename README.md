
# MCUBE Synthetic Data Generator

![MCUBE Banner](./Logo.png)

## 🚀 Overview

**MCUBE Synthetic Data Generator** is a desktop application designed for creating, editing, and augmenting synthetic image datasets with polygonal annotations.
It supports both YOLO and COCO annotation formats and offers a sleek, intuitive interface to streamline your workflow.
You can globally adjust image properties such as brightness, contrast, saturation, and sharpness, perform pixel-perfect copy/cut/paste operations on annotated regions, and apply precise transformations (rotate, scale, translate) centered on each object.
All edits and augmentations maintain visual consistency, ensuring your exported images and annotation files are immediately ready for use in computer vision training.
---

## 🎨 Screenshots

![Example 1](./frame-1%20(1).png) 
*Example 1:*

![Example 2](./frame-1%20(3).png)  
*Example 2:*

![Example 3](./frame-1%20(2).png)  
*Example 3:*
---

## ✨ Features

* **Intuitive GUI** for seamless image and annotation management
* **Copy, Cut, and Paste** polygonal instances with pixel-perfect accuracy
* **Random Variation Generator** to quickly create augmented dataset samples
* Supports both **YOLO** and **COCO** annotation formats
* Customizable **cut fill color** options
* Polygon transformation controls: **rotate**, **scale**, **translate**
* Instant preview of edits and updated annotation visualization

---

## 🖥️ Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/meehirmhatrepy/MCUBE-Synthetic-Data-Generator
   cd MCUBE-Synthetic-Data-Generator
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```sh
   python app.py
   ```

---

## 📸 Usage Instructions

### 1. Load an Image

* Click **Upload Image** and select your image file.

### 2. Load Annotations

* For YOLO: Click **Upload YOLO .txt** and select your segmentation file.
* For COCO: Click **Upload COCO .json** and select your annotation file.

### 3. Select & Edit Instances

* Click a polygon to select it.
* Use **Copy**, **Cut**, and **Paste** buttons to duplicate or move instances.
* Use **Transform Selected** sliders to rotate or scale the selected polygon.
* Click **Reset Transform** to undo changes.

### 4. Random Variations

* Choose a class from the dropdown menu.
* Set the desired number of variations.
* Click **Generate Random Variations** to create new augmented images and annotation files.

### 5. Export

* The default export folder is `generated images and annos` in the project directory.
* Click **Save Current (COCO .json)** to save the current image and annotations.

---


## ⚡ Keyboard Shortcuts

* **Ctrl + C**: Copy selected instance
* **Ctrl + X**: Cut selected instance
* **Ctrl + V**: Paste instance
* **Ctrl + Z**: Undo
* **Ctrl + Y**: Redo

---

## 📝 License

GNU General Public License v3.0 © 2025 MCUBE Team

---

## 🤝 Contributing

Pull requests and suggestions are welcome!
Please open an issue to report bugs or request features.

---

## 📧 Contact

For support or collaboration, email: [meehirmhatre1234@gmail.com](mailto:meehirmhatre1234@gmail.com)

---

> **MCUBE Synthetic Data Generator** — Make your vision data smarter.
