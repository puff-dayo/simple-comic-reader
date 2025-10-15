# 简单漫画阅读器

[English](README.md) | [简体中文](README.zh_CN.md) | [Русский](README.ru_RU.md)

一个使用 Qt for Python 构建的轻量级漫画与图像阅读器。  
支持查看漫画图片、ZIP/CBZ 压缩包和 PDF 文件，具备缩放、平移和全屏功能。

---

## 功能特性

- 文件系统导航
- PDF 渲染缓存
- 无需解压即可查看 `.zip` / `.cbz` 漫画
- 支持 `.jpg`、`.jpeg`、`.png`、`.gif`、`.bmp`、`.webp`
- 缩放与自适应模式
- 键盘快捷键
- 可配置设置（保存在 `config.ini`）
- 多语言界面（English / 简体中文 / Русский）

---

## 使用方法

下载二进制版本直接运行，或从源码构建。

---

## 构建方式

```bash
uv venv
uv sync
.venv\Scripts\Activate.ps1
build.bat
```

---

## 依赖项

Python 3.12+

```python
dependencies = [
    "nuitka>=2.8.1",
    "pillow>=11.3.0",
    "pymupdf==1.23.5",
    "pyside6>=6.10.0",
]
```