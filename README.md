# Simple Comic Reader

[English](README.md) | [简体中文](README.zh_CN.md) | [Русский](README.ru_RU.md)

A lightweight, comic and image reader built with Qt for Python.  
Viewing comic images, ZIP/CBZ archives, and PDF files — with zoom, pan, and fullscreen capabilities.

---

## Features

- File system navigation
- PDF render with cache
- View `.zip` / `.cbz` comic without extraction
- Support `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`
- Zoom & Fit Modes
- Keyboard Shortcuts
- Configurable Settings saved in `config.ini`
- Multi-language UI (English / 简体中文 / Русский)

---

##  Usage

Download binary release and run, or build from source.

---

## Build

```bash
uv venv
uv sync
.venv\Scripts\Activate.ps1
build.bat
```

---

## Dependencies

Python 3.12+


```python
dependencies = [
    "nuitka>=2.8.1",
    "pillow>=11.3.0",
    "pymupdf==1.23.5",
    "pyside6>=6.10.0",
]
```
