import cv2
import numpy as np
from typing import Tuple, Union
from PySide6.QtGui import QImage, QPixmap


def qimage_to_numpy_rgba(qimg: Union[QImage, QPixmap]) -> np.ndarray:
    if isinstance(qimg, QPixmap):
        qimg = qimg.toImage()

    img = qimg.convertToFormat(QImage.Format_RGBA8888)
    w = int(img.width())
    h = int(img.height())
    bpl = int(img.bytesPerLine())
    expected_size = bpl * h

    try:
        ptr = img.constBits()
    except Exception:
        try:
            ptr = img.bits()
        except Exception:
            ptr = None

    if ptr is None:
        raise RuntimeError("Cannot access QImage bits/buffer.")

    try:
        if isinstance(ptr, memoryview):
            buf = ptr.tobytes()
        else:
            try:
                buf = ptr.asstring(expected_size)
            except Exception:
                try:
                    mv = memoryview(ptr)
                    buf = mv.tobytes()
                except Exception:
                    buf_lines = []
                    for y in range(h):
                        line = img.scanLine(y)
                        try:
                            if isinstance(line, (bytes, bytearray, memoryview)):
                                buf_lines.append(bytes(line))
                            else:
                                buf_lines.append(line.asstring(bpl))
                        except Exception:
                            raise RuntimeError("Unable to read QImage scanLine buffers.")
                    buf = b"".join(buf_lines)
    except Exception as e:
        raise RuntimeError("Failed to read QImage bits buffer: " + str(e))

    if buf is None:
        raise RuntimeError("Failed to obtain raw bytes from QImage.")

    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size < expected_size:
        if arr.size == w * h * 4:
            arr = arr.reshape((h, w, 4)).copy()
            return arr
        raise RuntimeError(f"Buffer size {arr.size} smaller than expected {expected_size}.")

    arr = arr.reshape((h, bpl))
    arr = arr[:, :w * 4].reshape((h, w, 4)).copy()
    return arr


def numpy_to_qimage_rgb_or_rgba(img: np.ndarray) -> QImage:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim != 3:
        raise ValueError("Expected 3D array (H, W, C)")

    h, w, c = img.shape
    if c not in (3, 4):
        raise ValueError("Expected 3 (RGB) or 4 (RGBA) channels")

    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)

    if c == 3:
        bytes_per_line = w * 3
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:  # c == 4
        bytes_per_line = w * 4
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGBA8888)

    return qimg.copy()


def compute_local_std(gray: np.ndarray, k: int = 15) -> np.ndarray:
    scale = 0.5
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k_small = max(3, k // 2)

    mean = cv2.blur(small.astype(np.float32), (k_small, k_small))
    mean_sq = cv2.blur((small.astype(np.float32) ** 2), (k_small, k_small))
    var = mean_sq - mean * mean
    var[var < 0] = 0
    std_small = np.sqrt(var)

    return cv2.resize(std_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def make_line_texture_mask(gray: np.ndarray, std_k: int = 11, std_mult: float = 0.7) -> np.ndarray:
    std_map = compute_local_std(gray, k=std_k)
    med = float(np.median(std_map))
    std_thresh = max(1.0, med * std_mult)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = np.abs(lap)
    e_min, e_max = float(edge.min()), float(edge.max())
    edge_norm = (edge - e_min) / (e_max - e_min + 1e-12)

    line_mask = (edge_norm > 0.12) & (std_map < std_thresh)
    mask = (line_mask.astype(np.uint8) * 255)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    return mask


def antialiased_downscale(img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    in_h, in_w = img.shape[:2]
    out_w, out_h = out_size
    ratio = max(in_w / out_w, in_h / out_h)
    if ratio <= 1.0:
        return cv2.resize(img, out_size, interpolation=cv2.INTER_LANCZOS4)
    sigma = 0.6 * ratio
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.resize(blurred, out_size, interpolation=cv2.INTER_LANCZOS4)


def preserve_lines_resize_sharp(img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(img, out_size, interpolation=cv2.INTER_LANCZOS4)

    g1 = cv2.GaussianBlur(resized, (0, 0), sigmaX=0.6)
    unsharp1 = cv2.addWeighted(resized, 1.8, g1, -0.8, 0)

    lap = cv2.Laplacian(unsharp1, cv2.CV_32F, ksize=1)
    enhanced = unsharp1.astype(np.float32) + lap * 0.5

    enhanced_u8 = np.clip(enhanced, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE per channel if color, or directly if grayscale
    if enhanced_u8.ndim == 3:
        result = np.zeros_like(enhanced_u8)
        for i in range(enhanced_u8.shape[2]):
            result[:, :, i] = clahe.apply(enhanced_u8[:, :, i])
    else:
        result = clahe.apply(enhanced_u8)

    return result


def combined_rescale_color(img: np.ndarray, out_size: Tuple[int, int],
                           std_k: int = 11, std_mult: float = 0.7) -> np.ndarray:

    if img.ndim == 2:
        return combined_rescale_gray(img, out_size, std_k, std_mult)

    if img.ndim != 3:
        raise ValueError("Expected 2D (grayscale) or 3D (color) array")

    has_alpha = (img.shape[2] == 4)

    # Extract alpha
    if has_alpha:
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
    else:
        alpha = None
        rgb = img

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    mask = make_line_texture_mask(gray, std_k=std_k, std_mult=std_mult)
    mask_f = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0) / 255.0

    tex_resized = antialiased_downscale(rgb, out_size)
    line_resized = preserve_lines_resize_sharp(rgb, out_size)

    mask_out = cv2.resize(mask_f, out_size, interpolation=cv2.INTER_LINEAR)
    mask_out = np.clip(mask_out, 0.0, 1.0)

    mask_3d = mask_out[:, :, np.newaxis]

    combined_rgb = (mask_3d * line_resized + (1.0 - mask_3d) * tex_resized).astype(np.uint8)

    if has_alpha:
        alpha_resized = cv2.resize(alpha, out_size, interpolation=cv2.INTER_LINEAR)
        combined = np.dstack([combined_rgb, alpha_resized])
    else:
        combined = combined_rgb

    return combined


def combined_rescale_gray(gray: np.ndarray, out_size: Tuple[int, int],
                          std_k: int = 11, std_mult: float = 0.7) -> np.ndarray:
    mask = make_line_texture_mask(gray, std_k=std_k, std_mult=std_mult)
    mask_f = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0) / 255.0

    tex_resized = antialiased_downscale(gray, out_size)
    line_resized = preserve_lines_resize_sharp(gray, out_size)

    mask_out = cv2.resize(mask_f, out_size, interpolation=cv2.INTER_LINEAR)
    mask_out = np.clip(mask_out, 0.0, 1.0)

    combined = (mask_out * line_resized + (1.0 - mask_out) * tex_resized).astype(np.uint8)
    return combined


def rescale_qimage_comic(qimg: Union[QImage, QPixmap], target_w: int, target_h: int,
                         mode: str = "keep_aspect",
                         std_k: int = 11, std_mult: float = 0.7) -> QImage:
    # std_k: Kernel size for texture detection (higher = more aggressive smoothing)
    # std_mult: Texture threshold multiplier (lower = more areas treated as lines)

    if isinstance(qimg, QPixmap):
        qimg = qimg.toImage()

    src_w = int(qimg.width())
    src_h = int(qimg.height())

    if src_w <= 0 or src_h <= 0:
        return qimg

    if mode == "fit_width":
        new_w = max(1, int(target_w))
        new_h = max(1, int(round(src_h * (new_w / src_w))))
    elif mode == "fit_height":
        new_h = max(1, int(target_h))
        new_w = max(1, int(round(src_w * (new_h / src_h))))
    else:  # keep_aspect
        scale_w = float(target_w) / float(src_w)
        scale_h = float(target_h) / float(src_h)
        scale = min(scale_w, scale_h) if (scale_w > 0 and scale_h > 0) else max(scale_w, scale_h)
        if scale <= 0:
            scale = 1.0
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

    if new_w == src_w and new_h == src_h:
        return qimg

    arr_rgba = qimage_to_numpy_rgba(qimg)

    if np.all(arr_rgba[:, :, 3] == 255):
        rgb = arr_rgba[:, :, :3]
        has_alpha = False
    else:
        rgb = arr_rgba[:, :, :3]
        alpha = arr_rgba[:, :, 3]
        has_alpha = True
        img_with_alpha = arr_rgba

    if has_alpha:
        result = combined_rescale_color(img_with_alpha, (new_w, new_h), std_k=std_k, std_mult=std_mult)
    else:
        result = combined_rescale_color(rgb, (new_w, new_h), std_k=std_k, std_mult=std_mult)

    return numpy_to_qimage_rgb_or_rgba(result)


def rescale_opencv_array_comic(arr: np.ndarray, target_w: int, target_h: int,
                               mode: str = "keep_aspect",
                               std_k: int = 11, std_mult: float = 0.7) -> np.ndarray:

    src_h, src_w = arr.shape[:2]

    if mode == "fit_width":
        new_w = max(1, int(target_w))
        new_h = max(1, int(round(src_h * (new_w / src_w))))
    elif mode == "fit_height":
        new_h = max(1, int(target_h))
        new_w = max(1, int(round(src_w * (new_h / src_h))))
    else:  # keep_aspect
        scale_w = float(target_w) / float(src_w)
        scale_h = float(target_h) / float(src_h)
        scale = min(scale_w, scale_h) if (scale_w > 0 and scale_h > 0) else max(scale_w, scale_h)
        if scale <= 0:
            scale = 1.0
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

    if new_w == src_w and new_h == src_h:
        return arr

    return combined_rescale_color(arr, (new_w, new_h), std_k=std_k, std_mult=std_mult)