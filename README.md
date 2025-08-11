# MCUBE Synthetic Data Generator

![MCUBE Banner](./Logo.png)


## 🚀 Overview

**MCUBE Synthetic Data Generator** is a state-of-the-art, minimal desktop application for creating, editing, and augmenting synthetic image datasets with polygonal annotations. MCUBE supports both YOLO and COCO annotation formats, and features a sleek, intuitive interface for seamless workflow. You can globally enhance images (brightness, contrast, saturation, sharpness), perform pixel-perfect copy/cut/paste of annotated regions, and apply advanced transformations (rotate, scale, translate) centered on each object. All edits and augmentations are visually consistent, ensuring your exported images and annotations are ready for robust computer vision training.
---

## ✨ Features

- **Intuitive GUI** for image and annotation management
- **Copy/Cut/Paste** polygonal instances with pixel-perfect alignment
- **Random Variation Generator** for fast dataset augmentation
- **Supports YOLO & COCO formats**
- **Customizable cut fill color**
- **Polygon transformation controls** (rotate, scale, translate)
- **Instant preview and annotation visualization**

---

## 🖥️ Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-org/mcube-synthetic-data-generator.git
   cd mcube-synthetic-data-generator
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

### 1. **Load an Image**
- Click **Upload Image** and select your image file.

### 2. **Load Annotations**
- For YOLO: Click **Upload YOLO .txt** and select your segmentation file.
- For COCO: Click **Upload COCO .json** and select your annotation file.

### 3. **Select & Edit Instances**
- Click on a polygon to select it.
- Use **Copy**, **Cut**, and **Paste** buttons to duplicate or move instances.
- Use the **Transform Selected** sliders to rotate or scale the selected polygon.
- **Reset Transform** to revert changes.

### 4. **Random Variations**
- Choose a class from the dropdown.
- Set the number of variations.
- Click **Generate Random Variations** to create new images and annotations.

### 5. **Export**
- The default export folder is `generated images and annos` in your project directory.
- Click **Save Current (COCO .json)** to save the current image and annotations.

---

## 🎨 Screenshots

![Main UI](https://user-images.githubusercontent.com/your-ui-screenshot.png)
*Main interface with annotation and transformation controls.*

---

## ⚡ Keyboard Shortcuts

- **Ctrl+C**: Copy selected instance
- **Ctrl+X**: Cut selected instance
- **Ctrl+V**: Paste instance

---

## 💡 Tips

- Use **Cut Fill Color** to choose how cut regions are filled (black or mean color).
- All copy/paste and random generation operations use bounding box top-left for perfect alignment.

---

## 📝 License

MIT License © 2025 MCUBE Team

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Please open an issue for bug reports or feature requests.

---

## 📧 Contact

For support or collaboration, email: [meehirmhatre1234@gmail.com](mailto:meehirmhatre1234@gmail.com)

---

> **MCUBE Synthetic Data Generator** — Make your vision data smarter,