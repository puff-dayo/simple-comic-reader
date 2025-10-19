import cv2
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any, Iterable
from PySide6.QtGui import QImage, QPixmap
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

_executor = ThreadPoolExecutor(max_workers=8)


# -----------------------
# Utility helpers
# -----------------------
def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def _ensure_contiguous(img: np.ndarray) -> np.ndarray:
    if not img.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(img)
    return img


def _parallel_submit(callables: Iterable[Tuple[Any, tuple, dict]]) -> Dict[Future, str]:
    futures: Dict[Future, str] = {}
    for func, args, kwargs in callables:
        label = kwargs.pop('__label', None)
        fut = _executor.submit(func, *args, **kwargs)
        futures[fut] = label if label is not None else func.__name__
    return futures


def _compute_new_size(src_w: int, src_h: int, target_w: int, target_h: int, mode: str = "keep_aspect"):
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
    return new_w, new_h


# -----------------------
# Low-level image ops
# -----------------------
def antialiased_downscale(img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    img = _ensure_contiguous(img)
    in_h, in_w = img.shape[:2]
    out_w, out_h = out_size
    ratio = max(in_w / out_w, in_h / out_h)
    if ratio <= 1.0:
        return cv2.resize(img, out_size, interpolation=cv2.INTER_LANCZOS4)
    sigma = 0.6 * ratio
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.resize(blurred, out_size, interpolation=cv2.INTER_LANCZOS4)


def _prefilt_sigma_for_target_long(L1: int, L2: int, target_half_pixel: float = 0.5) -> float:
    R = float(L1) * float(target_half_pixel) / max(1.0, float(L2))
    return max(0.5, R)


def _gaussian_blur_with_sigma(img: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(6 * sigma) | 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


# -----------------------
# Texture / moiré / line detection
# -----------------------
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


def _fft_repeatiness_score(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = r2 > (min(h, w) * 0.05) ** 2
    m = mag * mask
    vals = m[mask]
    if vals.size == 0:
        return 0.0
    peak = vals.max()
    med = float(np.median(vals))
    return float(peak / (med + 1e-9))


def _make_moire_mask_from_rgb(rgb: np.ndarray, std_k: int = 11, std_mult: float = 0.7,
                              fft_threshold: float = 8.0, tile: int = 256,
                              fft_tile_min_size: int = 32) -> np.ndarray:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb.copy()

    try:
        spatial_mask = make_line_texture_mask(gray, std_k=std_k, std_mult=std_mult) / 255.0
    except Exception:
        spatial_mask = np.zeros_like(gray, dtype=np.float32)

    tiles = []
    h_tiles = (h + tile - 1) // tile
    w_tiles = (w + tile - 1) // tile
    for ty in range(h_tiles):
        y0 = ty * tile
        y1 = min(h, y0 + tile)
        for tx in range(w_tiles):
            x0 = tx * tile
            x1 = min(w, x0 + tile)
            ph = y1 - y0
            pw = x1 - x0
            if ph < fft_tile_min_size or pw < fft_tile_min_size:
                continue
            tiles.append((y0, y1, x0, x1))

    futures = {}
    for (y0, y1, x0, x1) in tiles:
        patch = gray[y0:y1, x0:x1].copy()
        fut = _executor.submit(_fft_repeatiness_score, patch)
        futures[fut] = (y0, y1, x0, x1)

    fft_mask = np.zeros_like(gray, dtype=np.float32)
    for fut in as_completed(futures):
        score = 0.0
        try:
            score = fut.result()
        except Exception:
            score = 0.0
        y0, y1, x0, x1 = futures[fut]
        if score >= fft_threshold:
            fft_mask[y0:y1, x0:x1] = 1.0

    combined = np.clip(spatial_mask + fft_mask, 0.0, 1.0)
    combined_u8 = (combined * 255.0).astype(np.uint8)
    combined_blur = cv2.GaussianBlur(combined_u8, (0, 0), sigmaX=4.0)
    combined_f = combined_blur.astype(np.float32) / 255.0
    return combined_f


# -----------------------
# Resizing strategies (public API)
# -----------------------
import cv2
import numpy as np
from typing import Tuple

def sharpen_edges(img: np.ndarray,
                  amount: float = 1.2,
                  radius: float = 0.8,
                  edge_thresh: float = 8.0,
                  mask_blur_sigma: float = 1.6,
                  morph_radius: int = 1,
                  method: str = "laplacian") -> np.ndarray:

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    is_color = (img.ndim == 3 and img.shape[2] >= 3)
    if is_color:
        bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        target_channel = L
    else:
        target_channel = img if img.ndim == 2 else img[:, :, 0]

    if method == "canny":
        edges = cv2.Canny(target_channel, max(1, edge_thresh - 2), edge_thresh * 2)
        mask = (edges > 0).astype(np.uint8) * 255
    else:
        lap = cv2.Laplacian(target_channel, cv2.CV_32F, ksize=3)
        edge_map = np.abs(lap)
        e_min, e_max = float(edge_map.min()), float(edge_map.max())
        if e_max - e_min < 1e-9:
            raw_mask = np.zeros_like(edge_map, dtype=np.float32)
        else:
            raw_mask = (edge_map - e_min) / (e_max - e_min)
        mask = (raw_mask > (edge_thresh / 100.0)).astype(np.uint8) * 255

    if morph_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_radius + 1, 2 * morph_radius + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=1)

    if mask_blur_sigma > 0.0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur_sigma, sigmaY=mask_blur_sigma)

    mask_f = (mask.astype(np.float32) / 255.0)

    sigma = max(0.3, float(radius))
    ksize = max(3, int(6 * sigma) | 1)
    blurred = cv2.GaussianBlur(target_channel, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(target_channel.astype(np.float32), 1.0 + amount,
                                blurred.astype(np.float32), -amount, 0.0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    out_channel = (mask_f * sharpened.astype(np.float32) + (1.0 - mask_f) * target_channel.astype(np.float32)).astype(np.uint8)

    if is_color:
        lab2 = cv2.merge([out_channel, A, B])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            return np.dstack([rgb2, alpha])
        return rgb2
    else:
        return out_channel


def preserve_lines_resize_sharp(img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    img = _ensure_contiguous(_ensure_uint8(img))
    resized = cv2.resize(img, out_size, interpolation=cv2.INTER_LANCZOS4)

    g1 = cv2.GaussianBlur(resized, (0, 0), sigmaX=0.6)
    unsharp1 = cv2.addWeighted(resized, 1.8, g1, -0.8, 0)

    lap = cv2.Laplacian(unsharp1, cv2.CV_32F, ksize=1)
    enhanced = unsharp1.astype(np.float32) + lap * 0.5

    enhanced_u8 = np.clip(enhanced, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    if enhanced_u8.ndim == 3 and enhanced_u8.shape[2] in (3, 4):
        has_alpha = (enhanced_u8.shape[2] == 4)
        rgb = enhanced_u8[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        if has_alpha:
            alpha = enhanced_u8[:, :, 3]
            return np.dstack([rgb2, alpha])
        return rgb2
    else:
        # grayscale
        return clahe.apply(enhanced_u8)


def _resize_and_prefilt_variants(rgb: np.ndarray, out_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    h, w = rgb.shape[:2]
    out_w, out_h = out_size
    sigma = _prefilt_sigma_for_target_long(max(h, w), max(out_h, out_w), target_half_pixel=0.5)
    rgb_prefilt = _gaussian_blur_with_sigma(rgb, float(sigma))

    callables = [
        (cv2.resize, (rgb, (out_w, out_h), cv2.INTER_LANCZOS4), {'__label': 'sharp'}),
        (cv2.resize, (rgb_prefilt, (out_w, out_h), cv2.INTER_AREA), {'__label': 'prefilt'}),
    ]
    futures = _parallel_submit(callables)

    results = {'sharp': None, 'prefilt': None}
    for fut, label in futures.items():
        try:
            res = fut.result()
        except Exception:
            res = None
        if label in results:
            results[label] = res

    if results['sharp'] is None:
        results['sharp'] = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    if results['prefilt'] is None:
        results['prefilt'] = cv2.resize(rgb_prefilt, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return results


def combined_rescale_color(img: np.ndarray, out_size: Tuple[int, int],
                           std_k: int = 11, std_mult: float = 0.7,
                           fft_threshold: float = 8.0, tile: int = 256,
                           fft_tile_min_size: int = 32) -> np.ndarray:

    if img.ndim == 2:
        return combined_rescale_gray(img, out_size, std_k=std_k, std_mult=std_mult,
                                     fft_threshold=fft_threshold, tile=tile,
                                     fft_tile_min_size=fft_tile_min_size)

    img = _ensure_uint8(_ensure_contiguous(img))
    h, w = img.shape[:2]
    out_w, out_h = out_size

    has_alpha = img.shape[2] == 4
    alpha = img[:, :, 3] if has_alpha else None
    rgb = img[:, :, :3] if has_alpha else img

    mask_orig = _make_moire_mask_from_rgb(rgb, std_k=std_k, std_mult=std_mult,
                                         fft_threshold=fft_threshold, tile=tile,
                                         fft_tile_min_size=fft_tile_min_size)

    variants = _resize_and_prefilt_variants(rgb, out_size)
    sharp_down = variants['sharp']
    prefilt_down = variants['prefilt']

    if has_alpha:
        alpha_down = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        alpha_down = None

    mask_out = cv2.resize((mask_orig * 255.0).astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask_out = cv2.GaussianBlur(mask_out, (0, 0), sigmaX=2.0)
    mask_f = (mask_out.astype(np.float32) / 255.0)[:, :, None]

    combined_rgb = np.clip(mask_f * prefilt_down.astype(np.float32) + (1.0 - mask_f) * sharp_down.astype(np.float32), 0, 255).astype(np.uint8)
    sharpened_rgb = sharpen_edges(combined_rgb,
                                  amount=1.0,
                                  radius=0.8,
                                  edge_thresh=9.0,
                                  mask_blur_sigma=1.6,
                                  morph_radius=1)

    if has_alpha:
        return np.dstack([sharpened_rgb, alpha_down])
    return sharpened_rgb


def combined_rescale_gray(gray: np.ndarray, out_size: Tuple[int, int],
                          std_k: int = 11, std_mult: float = 0.7,
                          fft_threshold: float = 8.0, tile: int = 256,
                          fft_tile_min_size: int = 32) -> np.ndarray:
    gray = _ensure_uint8(_ensure_contiguous(gray))
    h, w = gray.shape[:2]
    out_w, out_h = out_size

    mask_orig = _make_moire_mask_from_rgb(gray, std_k=std_k, std_mult=std_mult,
                                          fft_threshold=fft_threshold, tile=tile,
                                          fft_tile_min_size=fft_tile_min_size)

    sigma = _prefilt_sigma_for_target_long(max(h, w), max(out_h, out_w), target_half_pixel=0.5)
    gray_prefilt = _gaussian_blur_with_sigma(gray, float(sigma))

    callables = [
        (cv2.resize, (gray, (out_w, out_h), cv2.INTER_LANCZOS4), {'__label': 'sharp'}),
        (cv2.resize, (gray_prefilt, (out_w, out_h), cv2.INTER_AREA), {'__label': 'prefilt'}),
    ]
    futures = _parallel_submit(callables)

    sharp_down = prefilt_down = None
    for fut, label in futures.items():
        try:
            res = fut.result()
        except Exception:
            res = None
        if label == 'sharp':
            sharp_down = res
        elif label == 'prefilt':
            prefilt_down = res

    if sharp_down is None:
        sharp_down = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    if prefilt_down is None:
        prefilt_down = cv2.resize(gray_prefilt, (out_w, out_h), interpolation=cv2.INTER_AREA)

    mask_out = cv2.resize((mask_orig * 255.0).astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask_out = cv2.GaussianBlur(mask_out, (0, 0), sigmaX=2.0)
    mask_f = (mask_out.astype(np.float32) / 255.0)

    combined = np.clip(mask_f * prefilt_down.astype(np.float32) + (1.0 - mask_f) * sharp_down.astype(np.float32), 0, 255).astype(np.uint8)

    try:
        combined = sharpen_edges(combined,
                                 amount=1.0,
                                 radius=0.8,
                                 edge_thresh=9.0,
                                 mask_blur_sigma=1.6,
                                 morph_radius=1)
    except Exception as e:
        print("Warning: sharpen_edges failed:", e)

    return combined


# -----------------------
# QImage / numpy interop
# -----------------------
def qimage_to_numpy_rgba(qimg: Union[QImage, QPixmap]) -> np.ndarray:
    if isinstance(qimg, QPixmap):
        qimg = qimg.toImage()

    img = qimg.convertToFormat(QImage.Format_RGBA8888)
    w = int(img.width())
    h = int(img.height())
    bpl = int(img.bytesPerLine())
    expected_size = bpl * h

    ptr = None
    try:
        ptr = img.constBits()
    except Exception:
        try:
            ptr = img.bits()
        except Exception:
            ptr = None

    if ptr is None:
        raise RuntimeError("Cannot access QImage bits/buffer.")

    buf: Optional[bytes] = None
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
                    # fallback per-scanline
                    lines = []
                    for y in range(h):
                        line = img.scanLine(y)
                        try:
                            if isinstance(line, (bytes, bytearray, memoryview)):
                                lines.append(bytes(line))
                            else:
                                lines.append(line.asstring(bpl))
                        except Exception:
                            raise RuntimeError("Unable to read QImage scanLine buffers.")
                    buf = b"".join(lines)
    except Exception as e:
        raise RuntimeError("Failed to read QImage bits buffer: " + str(e))

    if buf is None:
        raise RuntimeError("Failed to obtain raw bytes from QImage.")

    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size < expected_size:
        if arr.size == w * h * 4:
            return arr.reshape((h, w, 4)).copy()
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

    img = _ensure_contiguous(img)

    if c == 3:
        bytes_per_line = w * 3
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:
        bytes_per_line = w * 4
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGBA8888)

    return qimg.copy()


# -----------------------
# High-level helpers
# -----------------------
def _run_two_branches_in_parallel(rgb: np.ndarray, out_size: Tuple[int, int]):
    rgb = _ensure_contiguous(rgb)

    fut_tex = _executor.submit(antialiased_downscale, rgb, out_size)
    fut_line = _executor.submit(preserve_lines_resize_sharp, rgb, out_size)

    try:
        tex_resized = fut_tex.result()
        line_resized = fut_line.result()
    except Exception:
        # best-effort cancel
        try:
            fut_tex.cancel()
            fut_line.cancel()
        except Exception:
            pass
        raise

    return tex_resized, line_resized


def rescale_qimage_comic(qimg: Union[QImage, QPixmap], target_w: int, target_h: int,
                         mode: str = "keep_aspect",
                         std_k: int = 11, std_mult: float = 0.7) -> QImage:
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
        # alpha = arr_rgba[:, :, 3]
        has_alpha = True
        img_with_alpha = arr_rgba

    if has_alpha:
        result = combined_rescale_color(img_with_alpha, (new_w, new_h), std_k=std_k, std_mult=std_mult)
    else:
        result = combined_rescale_color(rgb, (new_w, new_h), std_k=std_k, std_mult=std_mult)

    return numpy_to_qimage_rgb_or_rgba(result)
