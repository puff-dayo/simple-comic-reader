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
- 自动检测压缩包文件名编码格式
- 右键菜单可开启高质量去摩尔纹（De-moire）图像渲染算法
- 多语言界面（English / 简体中文 / Русский）
- 缩略图浏览功能：快速预览文件夹内所有漫画文件的封面
- 读取带密码的压缩包，在软件目录下创建 `pswd.txt`，每行一个密码

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
    "charset-normalizer>=3.4.4",
    "nuitka>=2.8.1",
    "numpy>=2.3.4",
    "opencv-python>=4.11.0.86",
    "pillow>=11.3.0",
    "pymupdf==1.23.5",
    "pyside6>=6.10.0",
    "pyzipper>=0.3.6",
]
```