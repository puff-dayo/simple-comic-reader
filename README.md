# Simple Comic Reader

[![CodeQL](https://github.com/puff-dayo/simple-comic-reader/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/puff-dayo/simple-comic-reader/actions/workflows/github-code-scanning/codeql) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/fbbf81f5e5434399bec0bc275ea988c9)](https://app.codacy.com/gh/puff-dayo/simple-comic-reader/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

[English](README.md) | [简体中文](README.zh_CN.md) | [Русский](README.ru_RU.md)

A lightweight, comic and image reader built with Qt for Python.  
Viewing comic images, ZIP/CBZ archives, and PDF files — with zoom, pan, and fullscreen capabilities.

<img width="520" height="auto" alt="" src="https://github.com/user-attachments/assets/b9b9bff6-1fcb-49e1-ae4c-f76cea602029" />

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

For Windows 64-bit, Download binary [release](https://github.com/puff-dayo/simple-comic-reader/releases) and run, or you can build from source for other OS.

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
