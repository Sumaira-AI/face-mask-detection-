"""Image enhancement / processing functions (all work on RGB uint8 numpy arrays)."""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def resize_image(img, scale=1.0):
    """Resize by a scale factor (e.g. 0.5 = half size)."""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)


def zoom_image(img, factor=1.0):
    """Zoom into the centre: crop centre region then resize back to original size."""
    if factor <= 1.0:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h / factor), int(w / factor)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    crop = img[y1:y1 + ch, x1:x1 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def gaussian_blur(img, k=5):
    return cv2.GaussianBlur(img, (_odd(k), _odd(k)), 0)


def average_blur(img, k=5):
    return cv2.blur(img, (_odd(k), _odd(k)))


def median_blur(img, k=5):
    return cv2.medianBlur(img, _odd(k))


def bilateral_filter(img, d=9, sigma=75):
    return cv2.bilateralFilter(img, int(d), sigma, sigma)


def to_grayscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def hist_equalization(img):
    """Equalize only the brightness (Y) channel so colours are not distorted."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def clahe(img, clip=2.0, grid=8):
    """Contrast Limited Adaptive Histogram Equalization (local contrast)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(int(grid), int(grid)))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def brightness_contrast(img, alpha=1.0, beta=0):
    """new = alpha*pixel + beta  (alpha=contrast, beta=brightness)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gamma_correction(img, gamma=1.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def sharpen(img, amount=1.0):
    """Unsharp masking: img + amount*(img - blurred)."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def canny_edges(img, low=100, high=200):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, int(low), int(high))
    return cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)


def sobel_edges(img, k=3):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=_odd(k))
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=_odd(k))
    mag = cv2.convertScaleAbs(np.sqrt(sx ** 2 + sy ** 2))
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)


def laplacian_edges(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F, ksize=3))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)


def negative(img):
    return 255 - img


def histogram_figure(img, title="Histogram"):
    """RGB channel histograms + grayscale histogram as a matplotlib figure."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    for i, c in enumerate(["red", "green", "blue"]):
        h = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax[0].plot(h, color=c, linewidth=1)
    ax[0].set_title(f"{title} - RGB")
    ax[0].set_xlim([0, 255])
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ax[1].hist(g.ravel(), bins=256, range=(0, 255), color="gray")
    ax[1].set_title(f"{title} - Grayscale")
    ax[1].set_xlim([0, 255])
    fig.tight_layout()
    return fig
