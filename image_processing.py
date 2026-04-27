"""
image_processing.py
===================
All image processing algorithms for the Image Processing App.
Each function takes a NumPy image array and returns a processed NumPy array.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import slic, felzenszwalb, active_contour
from skimage.filters import threshold_multiotsu
from skimage import color, filters
import heapq
from collections import Counter
import struct


# ─────────────────────────────────────────────────────────────
#  SECTION 1 ▸ LINEAR FILTERS
# ─────────────────────────────────────────────────────────────

def mean_box_filter(image, ksize=5):
    """Mean / Box Filter – replaces each pixel with the average of its neighbourhood."""
    return cv2.blur(image, (ksize, ksize))


def gaussian_filter(image, ksize=5, sigma=1.0):
    """Gaussian Filter – weighted average, reduces noise while preserving edges better than box."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def midpoint_filter(image, ksize=5):
    """Midpoint Filter – average of max and min pixel values in a neighbourhood."""
    from scipy.ndimage import maximum_filter, minimum_filter
    mx = maximum_filter(image, size=ksize)
    mn = minimum_filter(image, size=ksize)
    return ((mx.astype(np.float32) + mn.astype(np.float32)) / 2).astype(np.uint8)


def alpha_trimmed_mean_filter(image, ksize=5, d=2):
    """Alpha-Trimmed Mean Filter – trims d/2 highest and d/2 lowest values before averaging."""
    pad = ksize // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            if image.ndim == 3:
                for c in range(image.shape[2]):
                    vals = np.sort(region[:, :, c].flatten())
                    trimmed = vals[d//2: len(vals)-d//2]
                    result[i, j, c] = np.mean(trimmed)
            else:
                vals = np.sort(region.flatten())
                trimmed = vals[d//2: len(vals)-d//2]
                result[i, j] = np.mean(trimmed)
    return np.clip(result, 0, 255).astype(np.uint8)


def harmonic_mean_filter(image, ksize=3):
    """Harmonic Mean Filter – good for salt noise; formula: n / sum(1/pixel)."""
    pad = ksize // 2
    padded = cv2.copyMakeBorder(image.astype(np.float32) + 1e-6,
                                pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image, dtype=np.float32)
    n = ksize * ksize
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            result[i, j] = n / np.sum(1.0 / region, axis=(0, 1))
    return np.clip(result, 0, 255).astype(np.uint8)


def contraharmonic_mean_filter(image, ksize=3, Q=1.5):
    """Contraharmonic Mean Filter – Q>0 removes pepper noise, Q<0 removes salt noise."""
    pad = ksize // 2
    img_f = image.astype(np.float64) + 1e-6
    padded = cv2.copyMakeBorder(img_f, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            num = np.sum(region ** (Q + 1), axis=(0, 1))
            den = np.sum(region ** Q, axis=(0, 1)) + 1e-10
            result[i, j] = num / den
    return np.clip(result, 0, 255).astype(np.uint8)


def low_pass_filter(image, cutoff=30):
    """Low-Pass Filter (Frequency domain) – keeps low frequencies, removes high-freq noise."""
    def _lpf_channel(ch):
        dft = np.fft.fft2(ch)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 1, -1)
        filtered = dft_shift * mask
        back = np.fft.ifftshift(filtered)
        img_back = np.abs(np.fft.ifft2(back))
        return np.clip(img_back, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _lpf_channel(image)
    channels = [_lpf_channel(image[:, :, c]) for c in range(image.shape[2])]
    return cv2.merge(channels)


def high_pass_filter(image, cutoff=30):
    """High-Pass Filter (Frequency domain) – removes low frequencies, keeps edges/details."""
    def _hpf_channel(ch):
        dft = np.fft.fft2(ch)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 0, -1)
        filtered = dft_shift * mask
        back = np.fft.ifftshift(filtered)
        img_back = np.abs(np.fft.ifft2(back))
        return np.clip(img_back, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _hpf_channel(image)
    channels = [_hpf_channel(image[:, :, c]) for c in range(image.shape[2])]
    return cv2.merge(channels)


# ─────────────────────────────────────────────────────────────
#  SECTION 2 ▸ NON-LINEAR FILTERS
# ─────────────────────────────────────────────────────────────

def median_filter(image, ksize=5):
    """Median Filter – replaces pixel with neighbourhood median; excellent for salt-and-pepper noise."""
    return cv2.medianBlur(image, ksize if ksize % 2 == 1 else ksize + 1)


def mode_filter(image, ksize=5):
    """Mode Filter – replaces pixel with the most frequent value in neighbourhood."""
    from scipy.stats import mode as scipy_mode
    pad = ksize // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            if image.ndim == 3:
                for c in range(image.shape[2]):
                    vals = region[:, :, c].flatten()
                    m = scipy_mode(vals, keepdims=True)
                    result[i, j, c] = m.mode[0]
            else:
                vals = region.flatten()
                m = scipy_mode(vals, keepdims=True)
                result[i, j] = m.mode[0]
    return result


def maximum_filter_dilation(image, ksize=5):
    """Maximum Filter (Dilation) – replaces each pixel with neighbourhood maximum."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(image, kernel)


def minimum_filter_erosion(image, ksize=5):
    """Minimum Filter (Erosion) – replaces each pixel with neighbourhood minimum."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(image, kernel)


# ─────────────────────────────────────────────────────────────
#  SECTION 3 ▸ EDGE DETECTION
# ─────────────────────────────────────────────────────────────

def laplacian_filter(image):
    """Laplacian Filter – second-order derivative; detects edges in all directions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


def sobel_filter(image):
    """Sobel Filter – first-order gradient; detects horizontal and vertical edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx**2 + sy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def canny_edge_detector(image, low=50, high=150):
    """Canny Edge Detector – multi-stage algorithm; best general-purpose edge detector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Canny(gray, low, high)


def prewitt_filter(image):
    """Prewitt Filter – similar to Sobel but with uniform weights."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray_f = gray.astype(np.float64)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    gx = ndimage.convolve(gray_f, kx)
    gy = ndimage.convolve(gray_f, ky)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  SECTION 4 ▸ IMAGE SEGMENTATION
# ─────────────────────────────────────────────────────────────

def global_threshold(image, thresh=127):
    """Global Thresholding – simple binary split at a fixed intensity value."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return result


def otsu_threshold(image):
    """Otsu's Thresholding – automatically finds optimal global threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def adaptive_threshold(image, block_size=11, C=2):
    """Adaptive Thresholding – threshold varies across the image based on local area."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block_size, C)


def multi_otsu_threshold(image, classes=3):
    """Multi-Otsu Thresholding – extends Otsu to multiple classes/regions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    thresholds = threshold_multiotsu(gray, classes=classes)
    regions = np.digitize(gray, bins=thresholds)
    return (regions * (255 // (classes - 1))).astype(np.uint8)


def roberts_edge(image):
    """Roberts Edge Detection – cross-gradient operator; fast and simple."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray_f = gray.astype(np.float64)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    gx = ndimage.convolve(gray_f, kx)
    gy = ndimage.convolve(gray_f, ky)
    return np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)


def region_growing(image, seed=None):
    """Region Growing Segmentation – expands a region from a seed based on similarity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if seed is None:
        seed = (gray.shape[0] // 2, gray.shape[1] // 2)
    threshold = 15
    visited = np.zeros_like(gray, dtype=bool)
    result = np.zeros_like(gray)
    seed_val = int(gray[seed])
    stack = [seed]
    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        if abs(int(gray[y, x]) - seed_val) <= threshold:
            result[y, x] = 255
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < gray.shape[0] and 0 <= nx < gray.shape[1] and not visited[ny, nx]:
                    stack.append((ny, nx))
    return result


def split_and_merge(image):
    """Split and Merge – recursively splits regions then merges similar adjacent ones."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    result = np.zeros_like(gray)
    threshold = 10

    def split(y, x, h, w):
        region = gray[y:y+h, x:x+w]
        if region.size == 0:
            return
        if region.std() < threshold or min(h, w) <= 4:
            result[y:y+h, x:x+w] = region.mean()
            return
        h2, w2 = h // 2, w // 2
        split(y, x, h2, w2)
        split(y, x+w2, h2, w-w2)
        split(y+h2, x, h-h2, w2)
        split(y+h2, x+w2, h-h2, w-w2)

    split(0, 0, gray.shape[0], gray.shape[1])
    return result.astype(np.uint8)


def kmeans_segmentation(image, k=3):
    """K-Means Segmentation – clusters pixels into k colour groups."""
    data = image.reshape((-1, 3 if image.ndim == 3 else 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    return result.reshape(image.shape)


def watershed_segmentation(image):
    """Watershed Segmentation – treats image as topographic map; finds basin boundaries."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [0, 0, 255]
    return img_color


def slic_superpixels(image, n_segments=100, compactness=10):
    """SLIC Superpixels – groups nearby similar pixels into compact superpixel regions."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    segments = slic(img_rgb, n_segments=n_segments, compactness=compactness)
    result = color.label2rgb(segments, img_rgb, kind='avg')
    return cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def felzenszwalb_segmentation(image, scale=100, sigma=0.5, min_size=50):
    """Felzenszwalb Segmentation – graph-based method that merges similar regions."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    segments = felzenszwalb(img_rgb, scale=scale, sigma=sigma, min_size=min_size)
    result = color.label2rgb(segments, img_rgb, kind='avg')
    return cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────
#  SECTION 5 ▸ DATA COMPRESSION (Lossless)
# ─────────────────────────────────────────────────────────────

def rle_compress(image):
    """
    Run-Length Encoding (RLE) – encodes consecutive repeated pixel values.
    Returns a visualization showing compression ratio.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flat = gray.flatten()
    encoded = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i-1]:
            count += 1
        else:
            encoded.append((count, flat[i-1]))
            count = 1
    encoded.append((count, flat[-1]))
    original_size = len(flat)
    compressed_size = len(encoded) * 2
    ratio = original_size / max(compressed_size, 1)
    # Reconstruct decoded image for display
    decoded = np.array([val for (cnt, val) in encoded for _ in range(cnt)], dtype=np.uint8)
    result_img = decoded[:original_size].reshape(gray.shape)
    # Overlay compression info
    info = f"RLE Ratio: {ratio:.2f}x  ({original_size} -> {compressed_size} pairs)"
    result_color = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(result_color, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result_color


class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq


def huffman_compress(image):
    """
    Huffman Coding – builds optimal prefix-free code tree based on pixel frequency.
    Returns grayscale image with compression stats overlay.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flat = gray.flatten()
    freq = Counter(flat)
    heap = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        merged = HuffmanNode(None, l.freq + r.freq)
        merged.left, merged.right = l, r
        heapq.heappush(heap, merged)
    # Generate codes
    codes = {}
    def generate(node, code=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = code or "0"
            return
        generate(node.left, code + "0")
        generate(node.right, code + "1")
    if heap:
        generate(heap[0])
    total_bits = sum(len(codes[p]) * freq[p] for p in freq)
    original_bits = len(flat) * 8
    ratio = original_bits / max(total_bits, 1)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    info1 = f"Huffman: {original_bits} -> {total_bits} bits"
    info2 = f"Compression: {ratio:.2f}x | Symbols: {len(freq)}"
    cv2.putText(result_color, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(result_color, info2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return result_color


def lzw_compress(image):
    """
    LZW (Lempel-Ziv-Welch) – builds a dictionary of repeated patterns for compression.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flat = gray.flatten().tolist()
    # Build initial dictionary
    dict_size = 256
    dictionary = {(i,): i for i in range(dict_size)}
    w = (flat[0],)
    output = []
    for pixel in flat[1:]:
        wc = w + (pixel,)
        if wc in dictionary:
            w = wc
        else:
            output.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = (pixel,)
    output.append(dictionary[w])
    ratio = len(flat) / max(len(output), 1)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    info = f"LZW: {len(flat)} -> {len(output)} codes | Ratio: {ratio:.2f}x"
    cv2.putText(result_color, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result_color


def dpcm_compress(image):
    """
    Differential PCM (DPCM) – stores differences between consecutive pixels instead of absolute values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flat = gray.flatten().astype(np.int16)
    diff = np.diff(flat, prepend=flat[0])
    reconstructed = np.cumsum(diff).astype(np.int16)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8).reshape(gray.shape)
    # Show the difference signal normalized for display
    diff_visual = np.abs(diff).reshape(gray.shape)
    diff_visual = cv2.normalize(diff_visual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    result_color = cv2.cvtColor(diff_visual, cv2.COLOR_GRAY2BGR)
    non_zero = np.count_nonzero(diff)
    ratio = len(flat) / max(non_zero, 1)
    cv2.putText(result_color, f"DPCM Diff Map | Non-zero: {non_zero} | Ratio: {ratio:.2f}x",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return result_color


def arithmetic_compress(image):
    """
    Arithmetic Coding – encodes entire message as a single number; very high compression.
    Displays probability distribution visualization.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flat = gray.flatten()
    freq = Counter(flat.tolist())
    total = len(flat)
    entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
    theoretical_bits = entropy * total
    original_bits = total * 8
    ratio = original_bits / max(theoretical_bits, 1)
    # Visualize histogram as result
    hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
    max_count = max(freq.values())
    for val, cnt in freq.items():
        bar_height = int((cnt / max_count) * 180)
        cv2.line(hist_img, (val, 199), (val, 199 - bar_height), (100, 200, 100), 1)
    cv2.putText(hist_img, f"Arithmetic: Entropy={entropy:.2f} bits/px", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
    cv2.putText(hist_img, f"Theoretical ratio: {ratio:.2f}x | Symbols: {len(freq)}",
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    result = cv2.resize(hist_img, (gray.shape[1], gray.shape[0]))
    return result


# ─────────────────────────────────────────────────────────────
#  SECTION 6 ▸ IMAGE PADDING
# ─────────────────────────────────────────────────────────────

def _add_padding_label(image, label, pad):
    """Helper to add text label on padded image."""
    result = image.copy()
    cv2.putText(result, f"{label} (pad={pad})", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result


def zero_padding(image, pad=30):
    """Zero / Constant Padding – fills border with zeros (black pixels)."""
    result = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return _add_padding_label(result, "Zero Padding", pad)


def replicate_padding(image, pad=30):
    """Replicate / Edge Padding – repeats the edge pixels outward."""
    result = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return _add_padding_label(result, "Replicate Padding", pad)


def reflect_padding(image, pad=30):
    """Reflect Padding – mirrors pixels at the border (excludes border pixel)."""
    result = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    return _add_padding_label(result, "Reflect Padding", pad)


def symmetric_padding(image, pad=30):
    """Symmetric Padding – mirrors pixels at the border (includes border pixel)."""
    result = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    return _add_padding_label(result, "Symmetric Padding", pad)


def wrap_padding(image, pad=30):
    """Wrap / Periodic Padding – wraps the image as if it tiles periodically."""
    result = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_WRAP)
    return _add_padding_label(result, "Wrap Padding", pad)


def asymmetric_padding(image, top=10, bottom=40, left=20, right=60):
    """Asymmetric Padding – different padding on each side."""
    result = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    cv2.putText(result, f"Asymmetric (T{top} B{bottom} L{left} R{right})", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result


# ─────────────────────────────────────────────────────────────
#  BONUS ▸ EXTRA OPERATIONS
# ─────────────────────────────────────────────────────────────

def to_grayscale(image):
    """Convert image to grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def histogram_equalization(image):
    """Histogram Equalization – enhances contrast by redistributing intensities."""
    if image.ndim == 3:
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.equalizeHist(image)


def jpeg_compression(image, quality=30):
    """JPEG Compression – lossy compression using DCT; lower quality = smaller size."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)
