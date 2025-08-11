# MCUBE Synthetic Data Generator

![MCUBE Banner](./Logo.png)

## 🚀 Overview

**MCUBE Synthetic Data Generator** is a powerful desktop application for creating, editing, and augmenting **synthetic image datasets** with **polygonal annotations**.  
It supports both **YOLO** and **COCO** annotation formats, giving you full control over annotated objects — from **cut, copy, paste** to **rotation**, **scaling**, and **translation** — all with pixel-perfect accuracy.  

Whether you're building training datasets from scratch or boosting dataset diversity, MCUBE ensures **instant annotation updates** and visual consistency.  
With global image adjustments (brightness, contrast, saturation, sharpness) and an automated **Random Variation Generator**, your datasets are ready for training in minutes.

---

## 🎨 Screenshots

*Example 1:*  
![Example 1](./frame-1%20(1).png)  

*Example 2:*  
![Example 2](./frame-1%20(3).png)  

*Example 3:*  
![Example 3](./frame-1%20(2).png)  

---

## ✨ Features

* **Intuitive GUI** for seamless image and annotation management
* **Cut, Copy, and Paste** polygonal instances with pixel-perfect accuracy
* **Random Variation Generator** to quickly create diverse augmented samples
* Supports **YOLO** and **COCO** formats for import and export
* Customizable **cut fill color** options
* Polygon transformation controls: **rotate**, **scale**, **translate**
* **Live preview** of edits with instant annotation updates
* **Global image adjustments** for brightness, contrast, saturation, and sharpness

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
* Use **Copy**, **Cut**, and **Paste** to duplicate or move instances.
* Adjust **rotation** and **scale** using the **Transform Selected** sliders.
* Click **Reset Transform** to undo changes.

### 4. Random Variations

* Choose a class from the dropdown menu.
* Set the desired number of variations.
* Click **Generate Random Variations** to create new augmented images and updated annotation files.

### 5. Export

* Default export folder: `generated images and annos` in the project directory.
* Click **Save Current (COCO .json)** to save the current image and annotations.

---

## ⚡ Keyboard Shortcuts

* **Ctrl + C** — Copy selected instance
* **Ctrl + X** — Cut selected instance
* **Ctrl + V** — Paste instance
* **Ctrl + Z** — Undo
* **Ctrl + Y** — Redo

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

> \*\*MCUBE — Copy, Paste, Augment. Pixel-perfect synthetic dataset creation in minutes.
