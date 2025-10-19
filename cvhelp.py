import cv2
import numpy as np
from PySide6.QtGui import QImage
import traceback

from lib_cr import rescale_qimage_comic, _compute_new_size


def _dbg_print(*args, **kwargs):
    print("[resize_qimg_dbg]", *args, **kwargs)

def qimage_to_numpy_rgba(qimg: QImage) -> np.ndarray:
    try:
        img = qimg.convertToFormat(QImage.Format_RGBA8888)
    except Exception:
        img = qimg.convertToFormat(QImage.Format_ARGB32)

    h = img.height()
    w = img.width()
    if h == 0 or w == 0:
        return np.zeros((0, 0, 4), dtype=np.uint8)

    buf = img.constBits()

    try:
        arr = np.frombuffer(buf, dtype=np.uint8)
    except TypeError:
        arr = np.frombuffer(bytes(buf), dtype=np.uint8)

    row_bytes = int(img.bytesPerLine())
    expected = row_bytes * h

    if arr.size < expected:
        arr = np.frombuffer(bytes(buf), dtype=np.uint8)

    try:
        if row_bytes == w * 4:
            arr = arr.reshape((h, w, 4)).copy()
        else:
            out = np.empty((h, w, 4), dtype=np.uint8)
            for y in range(h):
                start = y * row_bytes
                row = arr[start:start + (w * 4)]
                out[y] = row.reshape((w, 4))
            arr = out
    except Exception:
        # Last-resort robust per-line copy
        _dbg_print("reshape per-line fallback")
        out = np.empty((h, w, 4), dtype=np.uint8)
        for y in range(h):
            start = y * row_bytes
            row = arr[start:start + (w * 4)]
            out[y] = row.reshape((w, 4))
        arr = out

    if arr.shape[2] == 4:
        # if alpha seems to be in channel 0 (ARGB), move channels to RGBA
        a_mean_channel0 = int(arr[:, :, 0].mean())
        a_mean_channel3 = int(arr[:, :, 3].mean())
        # If channel0 looks like alpha (lots of 255) and channel3 not, swap
        if a_mean_channel0 > a_mean_channel3 + 10:
            # currently [A, R, G, B] -> reorder to [R,G,B,A]
            arr = arr[:, :, [1, 2, 3, 0]].copy()

    return arr

def numpy_to_qimage_rgb_or_rgba(arr: np.ndarray) -> QImage:
    h, w = arr.shape[:2]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # Prefer RGBA8888
        try:
            return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        except Exception:
            rgba = arr.copy()
            bgra = rgba[:, :, [2, 1, 0, 3]].copy()
            return QImage(bgra.data, w, h, 4 * w, QImage.Format_ARGB32).copy()
    else:
        raise ValueError(f"Unsupported array shape for conversion to QImage: {arr.shape}")

def resize_qimage_with_opencv(qimage: QImage, target_w: int, target_h: int, mode: str = "keep_aspect",
                              use_comic_rescale: bool = False, *, std_k: int = 11, std_mult: float = 0.7) -> QImage:
    try:
        src_w = int(qimage.width())
        src_h = int(qimage.height())
        _dbg_print("ENTER resize: src:", src_w, src_h, "requested target:", target_w, target_h, "mode:", mode)

        if src_w <= 0 or src_h <= 0:
            _dbg_print("Invalid source size")
            return qimage

        new_w, new_h = _compute_new_size(src_w, src_h, target_w, target_h, mode)
        _dbg_print("Computed new size:", new_w, new_h)

        if new_w == src_w and new_h == src_h:
            _dbg_print("No-op resize (target equals source)")
            return qimage

        if use_comic_rescale:
            _dbg_print("Delegating to rescale_qimage_comic")
            result = rescale_qimage_comic(qimage, target_w, target_h, mode=mode, std_k=std_k, std_mult=std_mult)
            _dbg_print("Comic rescale done, result size:", result.width(), result.height())
            return result

        arr_rgba = qimage_to_numpy_rgba(qimage)
        _dbg_print("Converted to numpy arr shape:", arr_rgba.shape)

        cv_src = cv2.cvtColor(arr_rgba, cv2.COLOR_RGBA2BGRA)
        interp = cv2.INTER_AREA if (new_w < src_w or new_h < src_h) else cv2.INTER_LANCZOS4
        _dbg_print("Using interpolation:", "INTER_AREA" if interp == cv2.INTER_AREA else "LANCZOS4")
        cv_dst = cv2.resize(cv_src, (new_w, new_h), interpolation=interp)
        _dbg_print("cv2.resize done, result shape:", cv_dst.shape)
        rgba = cv2.cvtColor(cv_dst, cv2.COLOR_BGRA2RGBA)

        if np.all(rgba[:, :, 3] == 255):
            rgb = rgba[:, :, :3].copy()
            qout = numpy_to_qimage_rgb_or_rgba(rgb)
            _dbg_print("Alpha all 255 -> returned RGB QImage size:", qout.width(), qout.height())
            return qout
        else:
            qout = numpy_to_qimage_rgb_or_rgba(rgba)
            _dbg_print("Returned RGBA QImage size:", qout.width(), qout.height())
            return qout

    except Exception as e:
        _dbg_print("Exception in resize_qimage_with_opencv:", e)
        _dbg_print(traceback.format_exc())
        return qimage