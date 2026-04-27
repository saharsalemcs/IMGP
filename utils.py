"""
utils.py
========
Helper utilities: image loading, resizing, PIL↔CV2 conversion, saving.
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox


# ── Image I/O ─────────────────────────────────────────────────

def load_image(filepath: str) -> np.ndarray:
    """Load an image from disk using OpenCV (BGR format)."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Cannot load image: {filepath}")
    return img


def save_image(image: np.ndarray, filepath: str) -> bool:
    """Save an OpenCV image to disk. Returns True on success."""
    return cv2.imwrite(filepath, image)


def open_file_dialog() -> str:
    """Open a file dialog and return the selected image path."""
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
            ("All files", "*.*")
        ]
    )
    return path


def save_file_dialog() -> str:
    """Open a save dialog and return the chosen filepath."""
    path = filedialog.asksaveasfilename(
        title="Save Result Image",
        defaultextension=".png",
        filetypes=[
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("BMP", "*.bmp"),
        ]
    )
    return path


# ── Conversion helpers ────────────────────────────────────────

def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR/Gray image to PIL Image (RGB)."""
    if cv_image.ndim == 2:
        return Image.fromarray(cv_image)
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def cv2_to_tk(cv_image: np.ndarray, max_w: int = 500, max_h: int = 460) -> ImageTk.PhotoImage:
    """Convert an OpenCV image to a Tkinter-compatible PhotoImage, resized to fit a panel."""
    pil_img = cv2_to_pil(cv_image)
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


# ── Validation ────────────────────────────────────────────────

def ensure_image(image) -> bool:
    """Return True if image is a valid numpy array."""
    return image is not None and isinstance(image, np.ndarray) and image.size > 0


def show_error(title: str, message: str):
    """Show a Tkinter error dialog."""
    messagebox.showerror(title, message)


def show_info(title: str, message: str):
    """Show a Tkinter info dialog."""
    messagebox.showinfo(title, message)


# ── Image info ────────────────────────────────────────────────

def image_info(image: np.ndarray) -> str:
    """Return a short string describing image dimensions and channels."""
    if image is None:
        return "No image"
    h, w = image.shape[:2]
    ch = image.shape[2] if image.ndim == 3 else 1
    return f"{w}×{h} px  |  {ch} channel{'s' if ch > 1 else ''}  |  dtype: {image.dtype}"
