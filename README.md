# 🖼️ Image Processing Lab

A complete desktop image processing application built with **Python + Tkinter + OpenCV**.

---

## 📁 Project Structure

```
image_processing_app/
├── main.py               ← GUI application (run this)
├── image_processing.py   ← All 40+ algorithms
├── utils.py              ← Helpers (load/save/convert)
├── requirements.txt      ← Python dependencies
└── README.md
```

---

## ⚙️ Setup in VS Code (Step by Step)

### 1. Install Python
Download from https://www.python.org/downloads/ (Python 3.9 or newer).
Make sure to check **"Add Python to PATH"** during install.

### 2. Open the project in VS Code
```
File → Open Folder → select the image_processing_app folder
```

### 3. Open Terminal in VS Code
```
Terminal → New Terminal   (or Ctrl + `)
```

### 4. (Recommended) Create a virtual environment
```bash
python -m venv venv
```
Then activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Run the app
```bash
python main.py
```

---

## 🧩 Features

### Linear Filters
| Operation | Description |
|-----------|-------------|
| Mean / Box Filter | Averages neighbourhood pixels |
| Gaussian Filter | Weighted average, preserves edges |
| Midpoint Filter | Average of max and min |
| Alpha-Trimmed Mean | Trims extreme values before averaging |
| Harmonic Mean | Good for salt noise |
| Contraharmonic Mean | Good for pepper/salt noise |
| Low-Pass Filter | Frequency domain — removes high-freq noise |
| High-Pass Filter | Frequency domain — keeps edges/details |

### Non-Linear Filters
| Operation | Description |
|-----------|-------------|
| Median Filter | Removes salt-and-pepper noise |
| Mode Filter | Most frequent value in neighbourhood |
| Maximum (Dilation) | Brightens/expands objects |
| Minimum (Erosion) | Darkens/shrinks objects |

### Edge Detection
| Operation | Description |
|-----------|-------------|
| Laplacian | Second-order derivative |
| Sobel | Horizontal + vertical gradient |
| Canny | Multi-stage, best general edge detector |
| Prewitt | Similar to Sobel, uniform weights |
| Roberts | Cross-gradient, fast |

### Segmentation
| Operation | Description |
|-----------|-------------|
| Global Threshold | Fixed intensity split |
| Otsu Threshold | Automatic optimal threshold |
| Adaptive Threshold | Local varying threshold |
| Multi-Otsu | Multiple class thresholding |
| Region Growing | Expands from seed point |
| Split and Merge | Recursive region splitting |
| K-Means | Colour cluster grouping |
| Watershed | Topographic boundary detection |
| SLIC Superpixels | Compact superpixel grouping |
| Felzenszwalb | Graph-based region merging |

### Compression (Lossless)
| Operation | Description |
|-----------|-------------|
| RLE | Run-Length Encoding |
| Huffman Coding | Optimal prefix-free codes |
| LZW | Dictionary-based compression |
| DPCM | Differential encoding |
| Arithmetic Coding | Entropy visualization |
| JPEG | Lossy quality control |

### Padding
| Operation | Description |
|-----------|-------------|
| Zero Padding | Black border |
| Replicate / Edge | Repeats edge pixels |
| Reflect | Mirror at border (excludes edge) |
| Symmetric | Mirror at border (includes edge) |
| Wrap / Periodic | Tiles the image |
| Asymmetric | Different size on each side |

### Extra
| Operation | Description |
|-----------|-------------|
| Grayscale | Convert to grayscale |
| Histogram Equalization | Enhance contrast |

---

## 💡 How to Use

1. Click **Open Image** to load any JPG/PNG/BMP
2. Click a **category tab** (Linear Filters, Edge Detection, etc.)
3. Click an **operation** from the list
4. Adjust **parameters** with the sliders
5. Click **▶ Apply Operation**
6. View the result on the right panel
7. Click **Save Result** to export

---

## 📦 Dependencies

- `opencv-python` — image processing algorithms
- `numpy` — matrix operations
- `Pillow` — image display in Tkinter
- `scipy` — advanced filtering (mode, ndimage)
- `scikit-image` — SLIC, Felzenszwalb, Multi-Otsu, Active Contours

---

## 🛠️ Troubleshooting

**"No module named cv2"** → run `pip install opencv-python`

**"No module named skimage"** → run `pip install scikit-image`

**Tkinter not found on Linux** → run `sudo apt-get install python3-tk`

**Slow operations** (Mode Filter, Alpha-Trimmed on large images) → resize image first; these have O(n²) complexity per pixel.
