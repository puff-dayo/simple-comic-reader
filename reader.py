
import sys
import os
import zipfile
import configparser
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMenu, QMessageBox,
    QStyle, QSplitter, QPushButton, QComboBox, QScrollArea, QFileIconProvider
    )
from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QColorDialog, QDialogButtonBox, QKeySequenceEdit
from PySide6.QtGui import QKeySequence, QPixmap, QKeySequence, QShortcut, QAction, QCursor, QPalette, QIcon, QImage
from PySide6.QtCore import (
    Qt, QPoint, QRect, QSize, QEvent, QLocale, QFileInfo, Signal, QObject, QRunnable, QThreadPool, QTimer
    )

import fitz
import collections

import re

APP_VERSION = "1.1"

def is_archive_ext(ext: str) -> bool:
    if not ext:
        return False
    return ext.lower() in {'.zip', '.cbz'}

def is_archive_path_str(s: str) -> bool:
    try:
        return str(s).lower().endswith(('.zip', '.cbz'))
    except Exception:
        return False
    
def is_pdf_ext(ext: str) -> bool:
    if not ext:
        return False
    return ext.lower() == '.pdf'

def is_pdf_path_str(s: str) -> bool:
    try:
        return str(s).lower().endswith('.pdf')
    except Exception:
        return False

def is_pdf_protocol(s: str) -> bool:
    try:
        return isinstance(s, str) and s.startswith("pdf://")
    except Exception:
        return False

_num_re = re.compile(r'(\d+)')

def natural_sort_key(s):
    if isinstance(s, (Path, QFileInfo)):
        try:
            s = str(s)
        except Exception:
            s = s.name if hasattr(s, "name") else str(s)
    s = str(s)
    parts = _num_re.split(s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

    
icon_provider = QFileIconProvider()
_icon_cache = {}

def clear_icon_cache():
    global _icon_cache
    _icon_cache = {}

def get_icon_for_file(path: Path):
    try:
        ext = (path.suffix or "").lower()
    except Exception:
        ext = ""

    if ext in _icon_cache:
        return _icon_cache[ext]

    try:
        qi = QFileInfo(str(path))
        icon = icon_provider.icon(qi)
        if icon is None:
            icon = QIcon()
    except Exception:
        icon = QIcon()

    _icon_cache[ext] = icon
    return icon

CONFIG_PATH = Path(__file__).parent / "config.ini"

DEFAULT_CONFIG = {
    "general": {
        "auto_open_dir": "",
        "background_color": "auto",
        "default_fit_mode": "fit_page"
    },
    "shortcuts": {
        "prev_page": "Left",
        "next_page": "Right",
        "prev_archive": "Down",
        "next_archive": "Up",
        "reset_zoom": "Ctrl+0",
        "set_100": "Ctrl+1"
    }
}

UI_JSON = {
  "app": {
    "window_title": {
      "zh": "简单漫画阅读器",
      "en": "Simple Comic Reader",
      "ru": "Простой просмотрщик комиксов"
    },
    "about_owner": {
      "zh": "开发者：Setsuna (github@puffdayo)",
      "en": "Developer: Setsuna (github@puffdayo)",
      "ru": "Разработчик: Setsuna (github@puffdayo)"
    }
  },
  "buttons": {
    "open_folder": {
      "zh": "📁",
      "en": "📁",
      "ru": "📁"
    },
    "open_folder_tooltip": {
      "zh": "打开文件夹",
      "en": "Open folder",
      "ru": "Открыть папку"
    },
    "help": {
      "zh": "❔",
      "en": "❔",
      "ru": "❔"
    },
    "help_tooltip": {
      "zh": "查看帮助",
      "en": "Open help",
      "ru": "Открыть справку"
    },
    "settings": {
      "zh": "⚙️",
      "en": "⚙️",
      "ru": "⚙️"
    },
    "settings_tooltip": {
      "zh": "打开设置面板",
      "en": "Open settings",
      "ru": "Открыть настройки"
    },
    "fullscreen": {
      "zh": "⛶",
      "en": "⛶",
      "ru": "⛶"
    },
    "fullscreen_tooltip": {
      "zh": "切换全屏 (F11)",
      "en": "Toggle fullscreen (F11)",
      "ru": "Переключить полноэкранный режим (F11)"
    },
    "close_all": {
      "zh": "🧹",
      "en": "🧹",
      "ru": "🧹"
    },
    "close_all_tooltip": {
      "zh": "关闭所有已打开的压缩包",
      "en": "Close all opened archives",
      "ru": "Закрыть все открытые архивы"
    },
    "choose": {
      "zh": "选择...",
      "en": "Choose...",
      "ru": "Выбрать..."
    },
    "choose_color": {
      "zh": "选择颜色",
      "en": "Choose color",
      "ru": "Выбрать цвет"
    },
    "save": {
      "zh": "保存",
      "en": "Save",
      "ru": "Сохранить"
    },
    "cancel": {
      "zh": "取消",
      "en": "Cancel",
      "ru": "Отмена"
    }
  },
  "labels": {
    "file_list": {
      "zh": "文件列表",
      "en": "File list",
      "ru": "Список файлов"
    },
    "auto_open_dir": {
      "zh": "自动打开目录：",
      "en": "Auto-open directory:",
      "ru": "Каталог для автозагрузки:"
    },
    "bg_color": {
      "zh": "背景颜色 (#RRGGBB)：",
      "en": "Background color (#RRGGBB):",
      "ru": "Цвет фона (#RRGGBB):"
    },
    "default_fit": {
      "zh": "默认图像适配：",
      "en": "Default image fit:",
      "ru": "Режим масштабирования по умолчанию:"
    },
    "auto_dir_placeholder": {
      "zh": "启动时自动进入",
      "en": "Directory to open on startup",
      "ru": "Каталог, открываемый при старте"
    },
    "scale_combo": {
      "zh": "缩放模式选择",
      "en": "Scale mode",
      "ru": "Режим масштабирования"
    },
    "image_placeholder": {
      "zh": "选择图片查看",
      "en": "Select an image to view",
      "ru": "Выберите изображение для просмотра"
    }
  },
  "scale_options": {
    "fit_page": {
      "zh": "适应全页",
      "en": "Fit page",
      "ru": "По странице"
    },
    "fit_height": {
      "zh": "适应高",
      "en": "Fit height",
      "ru": "По высоте"
    },
    "fit_width": {
      "zh": "适应宽",
      "en": "Fit width",
      "ru": "По ширине"
    },
    "custom_percent": {
      "zh": "自定义 (%)",
      "en": "Custom (%)",
      "ru": "Пользовательский (%)"
    }
  },
  "shortcuts": {
    "prev_page": {
      "zh": "上一页",
      "en": "Previous page",
      "ru": "Предыдущая страница"
    },
    "next_page": {
      "zh": "下一页",
      "en": "Next page",
      "ru": "Следующая страница"
    },
    "prev_archive": {
      "zh": "上一个压缩包",
      "en": "Previous archive",
      "ru": "Предыдущий архив"
    },
    "next_archive": {
      "zh": "下一个压缩包",
      "en": "Next archive",
      "ru": "Следующий архив"
    },
    "reset_zoom": {
      "zh": "重置缩放",
      "en": "Reset zoom",
      "ru": "Сбросить масштаб"
    },
    "set_100": {
      "zh": "设为 100%",
      "en": "Set to 100%",
      "ru": "Установить 100%"
    }
  },
  "dialogs": {
    "help_html": {
      "zh": "<h3>📖 简单漫画阅读器</h3>\n"
            f"<p><b>版本：</b> {APP_VERSION}</p>\n"
            "<p><b>开发者：</b> Setsuna (github@puffdayo)</p>\n"
            "<hr>\n"
            "<p><b>使用说明：</b></p>\n"
            "<ul>\n"
            "<li><b>← / →</b>：上一页 / 下一页</li>\n"
            "<li><b>↑ / ↓</b>：上一个 / 下一个压缩包</li>\n"
            "<li><b>双击支持的文件</b>：展开查看或打开支持的文件</li>\n"
            "<li><b>右键</b>：显示操作功能选项 </li>\n"
            "<li><b>缩放模式：</b> 适应全页 / 适应高 / 适应宽 / 自定义百分比</li>\n"
            "<li><b>滚轮：</b> 当图片超出窗口时平移</li>\n"
            "<li><b>Ctrl + 滚轮：</b> 快速调整缩放（每格 ±10%）</li>\n"
            "<li><b>左右边缘点击：</b> 点击图片区域左右边缘可翻页</li>\n"
            "<li><b>F11 或 ⛶ 按钮：</b> 切换全屏 / 退出全屏</li>\n"
            "<li><b>隐藏文件面板：</b> 通过右键菜单或拖拽左右中间的分隔线至最左</li>\n"
            "<li><b>显示文件面板：</b> 通过右键菜单或从最左边缘拖拽分隔线向右</li>\n"
            "</ul>\n"
            "<h4>支持的文件类型</h4>\n"
            "<ul>\n"
            "<li>图片：.jpg, .jpeg, .png, .gif, .bmp</li>\n"
            "<li>压缩包：.zip, .cbz（双击可展开查看内部图片）</li>\n"
            "<li>PDF 文档：.pdf（逐页查看）</li>\n"
            "</ul>\n"
            "<hr>\n"
            "<p>程序记忆设置到 <code>config.ini</code> 文件中。</p>",
      "en": "<h3>📖 Simple Comic Reader</h3>\n"
            f"<p><b>Version:</b> {APP_VERSION}</p>\n"
            "<p><b>Developer:</b> Setsuna (github@puffdayo)</p>\n"
            "<hr>\n"
            "<p><b>Usage:</b></p>\n"
            "<ul>\n"
            "<li><b>← / →</b>: Previous / Next page</li>\n"
            "<li><b>↑ / ↓</b>: Previous / Next archive</li>\n"
            "<li><b>Double-click supported files</b>: Expand or open supported files</li>\n    "
            "<li><b>Right-click</b>: Show action menu</li>\n"
            "<li><b>Scale modes:</b> Fit page / Fit height / Fit width / Custom %</li>\n"
            "<li><b>Wheel:</b> Pan when image exceeds window</li>\n"
            "<li><b>Ctrl + Wheel:</b> Adjust zoom quickly (±10% per notch)</li>\n"
            "<li><b>Edge clicks:</b> Click left/right edges to flip pages</li>\n"
            "<li><b>F11 or ⛶ button:</b> Toggle fullscreen</li>\n"
            "<li><b>Hide file panel:</b> Use context menu or drag splitter left</li>\n"
            "<li><b>Show file panel:</b> Drag splitter right from far left</li>\n"
            "</ul>\n"
            "<h4>Supported file types</h4>\n"
            "<ul>\n"
            "<li>Images: .jpg, .jpeg, .png, .gif, .bmp</li>\n"
            "<li>Archives: .zip, .cbz (double-click to expand and view contained images)</li>\n"
            "<li>PDF: .pdf (view page-by-page)</li>\n"
            "</ul>\n"
            "<hr>\n"
            "<p>Program stores settings to <code>config.ini</code>.</p>",
      "ru": "<h3>📖 Простой просмотрщик комиксов</h3>\n"
            f"<p><b>Версия:</b> {APP_VERSION}</p>\n"
            "<p><b>Разработчик:</b> Setsuna (github@puffdayo)</p>\n"
            "<hr>\n"
            "<p><b>Инструкция:</b></p>\n"
            "<ul>\n"
            "<li><b>← / →</b>: Предыдущая / Следующая страница</li>\n"
            "<li><b>↑ / ↓</b>: Предыдущий / Следующий архив</li>\n"
            "<li><b>Двойной клик по поддерживаемым файлам</b>: открыть или развернуть поддерживаемые файлы</li>\n"
            "<li><b>Правый клик</b>: Показать меню действий</li>\n"
            "<li><b>Режимы масштабирования:</b> По странице / По высоте / По ширине / Процент</li>\n"
            "<li><b>Колесо мыши:</b> Панорамирование, когда изображение больше окна</li>\n"
            "<li><b>Ctrl + Колесо:</b> Быстрое изменение масштаба (±10% за шаг)</li>\n"
            "<li><b>Клики по краям:</b> Нажатие слева/справа для перелистывания</li>\n"
            "<li><b>F11 или ⛶:</b> Переключение полноэкранного режима</li>\n"
            "<li><b>Скрыть панель файлов:</b> Через контекстное меню или переместить разделитель влево</li>\n"
            "<li><b>Показать панель файлов:</b> Переместить разделитель вправо от самого левого края</li>\n"
            "</ul>\n"
            "<h4>Поддерживаемые типы файлов</h4>\n"
            "<ul>\n"
            "<li>Изображения: .jpg, .jpeg, .png, .gif, .bmp</li>\n"
            "<li>Архивы: .zip, .cbz (двойной клик — раскрыть и просмотреть файлы внутри)</li>\n"
            "<li>PDF: .pdf (просмотр по страницам)</li>\n"
            "</ul>\n"
            "<hr>\n"
            "<p>Программа сохраняет настройки в файле <code>config.ini</code>.</p>"
    },
    "settings_title": {
      "zh": "设置",
      "en": "Settings",
      "ru": "Настройки"
    },
    "about_title": {
      "zh": "关于 简单漫画阅读器",
      "en": "About Simple Comic Reader",
      "ru": "О Программе"
    },
    "help_title": {
      "zh": "帮助",
      "en": "Help",
      "ru": "Справка"
    },
    "info_saved": {
      "zh": "设置已保存",
      "en": "Settings saved",
      "ru": "Настройки сохранены"
    },
    "info_saved_details": {
      "zh": "设置已保存并应用。",
      "en": "Settings have been saved and applied.",
      "ru": "Настройки сохранены и применены."
    },
    "info_close_archives": {
      "zh": "所有已打开的压缩包已关闭。",
      "en": "All opened archives have been closed.",
      "ru": "Все открытые архивы закрыты."
    },
    "warning_zip_failed": {
      "zh": "解压失败: {error}",
      "en": "Failed to extract: {error}",
      "ru": "Не удалось распаковать: {error}"
    },
    "warning_load_failed": {
      "zh": "加载图片失败: {error}",
      "en": "Failed to load image: {error}",
      "ru": "Не удалось загрузить изображение: {error}"
    },
    "warning_save_failed": {
      "zh": "保存设置失败: {error}",
      "en": "Failed to save settings: {error}",
      "ru": "Не удалось сохранить настройки: {error}"
    },
    "info_clipboard_empty": {
      "zh": "当前没有可复制的图片。",
      "en": "No image available to copy.",
      "ru": "Нет изображения для копирования."
    },
    "info_copied": {
      "zh": "图片已复制到剪贴板。",
      "en": "Image copied to clipboard.",
      "ru": "Изображение скопировано в буфер обмена."
    }
  },
  "context_menu": {
    "copy_image": {
      "zh": "复制图片",
      "en": "Copy image",
      "ru": "Копировать изображение"
    },
    "show_hide_file_panel": {
      "zh": "显隐文件面板",
      "en": "Toggle file panel",
      "ru": "Показать/скрыть панель файлов"
    },
    "fit_page": {
      "zh": "适应全页",
      "en": "Fit page",
      "ru": "По странице"
    },
    "fit_height": {
      "zh": "适应高",
      "en": "Fit height",
      "ru": "По высоте"
    },
    "fit_width": {
      "zh": "适应宽",
      "en": "Fit width",
      "ru": "По ширине"
    },
    "reset_zoom": {
      "zh": "重置缩放",
      "en": "Reset zoom",
      "ru": "Сбросить масштаб"
    },
    "percent_options": {
      "zh": "百分比：50%, 75%, 100%, 125%, 150%, 200%",
      "en": "Percent options: 50%, 75%, 100%, 125%, 150%, 200%",
      "ru": "Проценты: 50%, 75%, 100%, 125%, 150%, 200%"
    },
    "previous": {
      "zh": "上一页",
      "en": "Previous",
      "ru": "Предыдущая"
    },
    "next": {
      "zh": "下一页",
      "en": "Next",
      "ru": "Следующая"
    },
    "sort_by_date": {
      "zh": "按日期排序",
      "en": "Sort by date",
      "ru": "Сортировать по дате"
    }
  },
  "tree": {
    "expand_zip_prefix": {
      "zh": "展开: ",
      "en": "Expanded: ",
      "ru": "Открыто: "
    },
    "zip_icon_tooltip": {
      "zh": "双击 ZIP 文件：展开查看内部图片",
      "en": "Double-click ZIP to expand and view contained images",
      "ru": "Двойной клик по ZIP — открыть и просмотреть файлы внутри"
    }
  },
  "edge_click": {
    "left_area_tooltip": {
      "zh": "点击左侧翻页（上一页）",
      "en": "Click left edge to go to previous page",
      "ru": "Клик слева — предыдущая страница"
    },
    "right_area_tooltip": {
      "zh": "点击右侧翻页（下一页）",
      "en": "Click right edge to go to next page",
      "ru": "Клик справа — следующая страница"
    }
  },
  "messages": {
    "no_selection": {
      "zh": "未选择任何文件",
      "en": "No file selected",
      "ru": "Файл не выбран"
    },
    "invalid_color": {
      "zh": "无效的颜色值，使用默认背景。",
      "en": "Invalid color value, using default background.",
      "ru": "Недопустимое значение цвета, используется фон по умолчанию."
    },
    "config_written_default": {
      "zh": "已写入默认配置到 config.ini",
      "en": "Default configuration written to config.ini",
      "ru": "Файл config.ini создан с настройками по умолчанию"
    }
  },
  "placeholders_and_helpers": {
    "help_usage_shortcuts": {
      "zh": "← / →：上一页 / 下一页；↑ / ↓：上一个 / 下一个压缩包；F11：全屏",
      "en": "← / →: prev/next page; ↑ / ↓: prev/next archive; F11: fullscreen",
      "ru": "← / →: предыдущая/следующая страница; ↑ / ↓: предыдущий/следующий архив; F11: полноэкранный"
    },
    "help_note": {
      "zh": "程序记忆设置到 config.ini 文件中。",
      "en": "Program stores settings in config.ini.",
      "ru": "Программа сохраняет настройки в config.ini."
    }
  }
}

def get_system_lang():
    loc = QLocale.system()

    primary = ""
    try:
        if hasattr(loc, "uiLanguages"):
            langs = loc.uiLanguages()
            if langs:
                primary = langs[0].split("-")[0].lower()
    except Exception:
        primary = ""

    if not primary:
        try:
            name = loc.name() or ""
            primary = name.split("_")[0].lower()
        except Exception:
            primary = ""

    if not primary:
        try:
            lang_enum = loc.language()
            if lang_enum == QLocale.Chinese:
                primary = "zh"
            elif lang_enum == QLocale.Russian:
                primary = "ru"
            elif lang_enum == QLocale.English:
                primary = "en"
        except Exception:
            primary = ""

    if primary.startswith("zh"):
        return "zh"
    if primary.startswith("ru"):
        return "ru"
    if primary.startswith("en"):
        return "en"
    return "en"


lang = get_system_lang()

def load_lang_from_JSON(lang):
    flat_ui = {}

    def is_translation_node(node):
        if not isinstance(node, dict):
            return False
        return all(isinstance(v, str) for v in node.values())

    def choose_translation(node_dict):
        for key in (lang, "ru", "zh", "en"):
            if key and key in node_dict and isinstance(node_dict[key], str):
                return node_dict[key]
        for v in node_dict.values():
            if isinstance(v, str):
                return v
        return ""

    def recurse(prefix, node):
        if isinstance(node, dict):
            if is_translation_node(node):
                flat_ui[prefix] = choose_translation(node)
            else:
                for k, v in node.items():
                    new_prefix = f"{prefix}_{k}" if prefix else k
                    recurse(new_prefix, v)
        else:
            flat_ui[prefix] = str(node)

    recurse("", UI_JSON)
    return flat_ui

UI = load_lang_from_JSON(lang)

class SettingsDialog(QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle(UI['dialogs_settings_title'])
        self.setModal(True)
        self.resize(320, 240)
        self.config = config or {}

        layout = QFormLayout(self)

        self.auto_dir_edit = QLineEdit(self)
        self.auto_dir_edit.setPlaceholderText(UI['labels_auto_dir_placeholder'])

        browse_btn = QPushButton(UI['buttons_choose'])
        def pick_dir():
            d = QFileDialog.getExistingDirectory(self, UI['labels_auto_open_dir'])
            if d:
                self.auto_dir_edit.setText(d)
        browse_btn.clicked.connect(pick_dir)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.auto_dir_edit)
        row_layout.addWidget(browse_btn)

        layout.addRow(QLabel(UI['labels_auto_open_dir']))
        layout.addRow(row_widget)

        self.bg_color_edit = QLineEdit(self)
        color_btn = QPushButton(UI['buttons_choose_color'])
        def pick_color():
            col = QColorDialog.getColor()
            if col.isValid():
                hexc = col.name()
                self.bg_color_edit.setText(hexc)
        color_btn.clicked.connect(pick_color)
        color_row = QWidget()
        color_row_layout = QHBoxLayout(color_row)
        color_row_layout.setContentsMargins(0,0,0,0)
        color_row_layout.addWidget(self.bg_color_edit)
        color_row_layout.addWidget(color_btn)
        layout.addRow(UI['labels_bg_color'], color_row)


        self.fit_combo = QComboBox(self)
        self.fit_combo.addItems([UI['scale_options_fit_page'], 
                                UI['scale_options_fit_height'],
                                UI['scale_options_fit_width']])
        layout.addRow(UI['labels_default_fit'], self.fit_combo)

        self.ks_prev = QKeySequenceEdit(self)
        self.ks_next = QKeySequenceEdit(self)
        self.ks_prev_arch = QKeySequenceEdit(self)
        self.ks_next_arch = QKeySequenceEdit(self)
        self.ks_reset = QKeySequenceEdit(self)
        self.ks_100 = QKeySequenceEdit(self)

        layout.addRow(UI['shortcuts_prev_page'], self.ks_prev)
        layout.addRow(UI['shortcuts_next_page'], self.ks_next)
        layout.addRow(UI['shortcuts_prev_archive'], self.ks_prev_arch)
        layout.addRow(UI['shortcuts_next_archive'], self.ks_next_arch)
        layout.addRow(UI['shortcuts_reset_zoom'], self.ks_reset)
        layout.addRow(UI['shortcuts_set_100'], self.ks_100)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.populate_from_config(self.config)

    def populate_from_config(self, cfg):
        gen = cfg.get("general", {})
        sc = cfg.get("shortcuts", {})
        self.auto_dir_edit.setText(gen.get("auto_open_dir", ""))
        self.bg_color_edit.setText(gen.get("background_color", "#ffffff"))  # auto
        mode = gen.get("default_fit_mode", "fit_page")
        if mode == "fit_page":
            self.fit_combo.setCurrentIndex(0)
        elif mode == "fit_height":
            self.fit_combo.setCurrentIndex(1)
        else:
            self.fit_combo.setCurrentIndex(2)

        self.ks_prev.setKeySequence(QKeySequence(sc.get("prev_page", "Left")))
        self.ks_next.setKeySequence(QKeySequence(sc.get("next_page", "Right")))
        self.ks_prev_arch.setKeySequence(QKeySequence(sc.get("prev_archive", "Down")))
        self.ks_next_arch.setKeySequence(QKeySequence(sc.get("next_archive", "Up")))
        self.ks_reset.setKeySequence(QKeySequence(sc.get("reset_zoom", "Ctrl+0")))
        self.ks_100.setKeySequence(QKeySequence(sc.get("set_100", "Ctrl+1")))

    def to_config(self):
        cfg = {"general": {}, "shortcuts": {}}
        cfg["general"]["auto_open_dir"] = self.auto_dir_edit.text().strip()
        cfg["general"]["background_color"] = self.bg_color_edit.text().strip() or "#ffffff"
        idx = self.fit_combo.currentIndex()
        cfg["general"]["default_fit_mode"] = ("fit_page", "fit_height", "fit_width")[idx]

        cfg["shortcuts"]["prev_page"] = self.ks_prev.keySequence().toString()
        cfg["shortcuts"]["next_page"] = self.ks_next.keySequence().toString()
        cfg["shortcuts"]["prev_archive"] = self.ks_prev_arch.keySequence().toString()
        cfg["shortcuts"]["next_archive"] = self.ks_next_arch.keySequence().toString()
        cfg["shortcuts"]["reset_zoom"] = self.ks_reset.keySequence().toString()
        cfg["shortcuts"]["set_100"] = self.ks_100.keySequence().toString()
        return cfg


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundRole(QPalette.Base)

        self._dragging = False
        self._last_pos = QPoint()
        self.drag_enabled = False
        self.scroll_area = None
        self.allow_pan_x = False
        self.allow_pan_y = False

        self.main_window = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drag_enabled and self.scroll_area:
            if not (self.allow_pan_x or self.allow_pan_y):
                return super().mousePressEvent(event)
            self._dragging = True
            self._last_pos = event.globalPosition().toPoint()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self.scroll_area:
            cur = event.globalPosition().toPoint()
            delta = cur - self._last_pos
            self._last_pos = cur

            hbar = self.scroll_area.horizontalScrollBar()
            vbar = self.scroll_area.verticalScrollBar()

            if self.allow_pan_x:
                hbar.setValue(hbar.value() - delta.x())
            if self.allow_pan_y:
                vbar.setValue(vbar.value() - delta.y())

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            if self.drag_enabled:
                self.setCursor(QCursor(Qt.OpenHandCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if (event.modifiers() & Qt.ControlModifier) and self.main_window:
            delta = event.angleDelta().y()
            if delta != 0:
                self.main_window.adjust_zoom_from_wheel(delta)
                event.accept()
                return
        super().wheelEvent(event)

    def contextMenuEvent(self, event):
        mw = self.main_window
        menu = QMenu(self)
        if mw:
            copy_act = QAction(UI['context_menu_copy_image'], self)
            copy_act.triggered.connect(mw.copy_current_to_clipboard)
            menu.addAction(copy_act)

            menu.addSeparator()

            fit_page_act = QAction(UI['context_menu_fit_page'], self)
            fit_page_act.triggered.connect(lambda: mw.set_scale_mode("fit_page"))
            menu.addAction(fit_page_act)

            fit_height_act = QAction(UI['context_menu_fit_height'], self)
            fit_height_act.triggered.connect(lambda: mw.set_scale_mode("fit_height"))
            menu.addAction(fit_height_act)

            fit_width_act = QAction(UI['context_menu_fit_width'], self)
            fit_width_act.triggered.connect(lambda: mw.set_scale_mode("fit_width"))
            menu.addAction(fit_width_act)

            menu.addSeparator()

            for pct in (50, 75, 100, 125, 150, 200):
                act = QAction(f"{pct}%", self)
                act.triggered.connect(lambda checked=False, p=pct: mw.set_custom_zoom(p))
                menu.addAction(act)

            menu.addSeparator()
            reset_act = QAction(UI['shortcuts_reset_zoom'], self)
            reset_act.triggered.connect(lambda: mw.set_scale_mode("fit_page"))
            menu.addAction(reset_act)
            
            # TODO: translation
            clear_cache_act = QAction("Clear render cache and reload", self)
            clear_cache_act.triggered.connect(lambda: mw.clear_cache_and_rerender())
            menu.addAction(clear_cache_act)

            toggle_list_act = QAction(UI['context_menu_show_hide_file_panel'], self)
            toggle_list_act.triggered.connect(mw.toggle_file_list)
            menu.addAction(toggle_list_act)

        menu.exec(event.globalPos())


class EdgeClickArea(QWidget):
    def __init__(self, parent, direction="left", callback=None, fixed_width=80, percent_width=None):
        super().__init__(parent)
        self.direction = direction
        self.callback = callback
        self.fixed_width = int(fixed_width)
        self.percent_width = float(percent_width) if percent_width is not None else None


        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setCursor(QCursor(Qt.PointingHandCursor))


        self.setMouseTracking(True)
        self.show()

    def sizeHint(self):
        parent = self.parent()
        if parent:

            w = self._computed_width(parent.width())
            return QSize(w, parent.height())
        return QSize(self.fixed_width, 200)

    def _computed_width(self, parent_width):
        if self.percent_width:
            w = max(1, int(round(parent_width * self.percent_width)))
            return w
        return self.fixed_width

    def set_percent_width(self, percent: float):
        self.percent_width = float(percent)
        if self.parent():
            self.setFixedWidth(self._computed_width(self.parent().width()))

    def paintEvent(self, event):

        return

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.callback:
            try:
                self.callback()
            except Exception:
                pass
            event.accept()
        else:
            super().mousePressEvent(event)


class _RenderSignal(QObject):
    finished = Signal(str, int, QPixmap)


class _PdfRenderTask(QRunnable):
    def __init__(self, pdf_path: str, page_num: int, target_w: int, target_h: int, sig: _RenderSignal):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.target_w = target_w
        self.target_h = target_h
        self.sig = sig

    def run(self):
        try:
            doc = fitz.open(self.pdf_path)
            page = doc[self.page_num]
            rect = page.rect
            orig_w, orig_h = rect.width, rect.height
            if orig_w <= 0 or orig_h <= 0:
                mat = fitz.Matrix(1, 1)
            else:
                scale = min(max(0.5, min(self.target_w / orig_w, self.target_h / orig_h)), 2.0)
                mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            n = pix.n
            if n == 3:
                fmt = QImage.Format_RGB888
            elif n == 4:
                fmt = QImage.Format_RGBA8888
            else:
                fmt = QImage.Format_RGB888
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
            qpix = QPixmap.fromImage(qimg)
            self.sig.finished.emit(self.pdf_path, self.page_num, qpix)
            doc.close()
        except Exception as e:
            try:
                self.sig.finished.emit(self.pdf_path, self.page_num, QPixmap())
            except Exception:
                pass


class ComicReader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI["app_window_title"])
        self.setWindowIcon(QIcon('icon-512.png'))
        self.setGeometry(100, 100, 1200, 800)

        self.config = {"general": dict(DEFAULT_CONFIG["general"]), "shortcuts": dict(DEFAULT_CONFIG["shortcuts"])}
        self.current_dir = Path()
        self.sort_by_date = False
        self.image_list = []
        self.current_index = 0

        self.current_zip_obj = None
        self.current_zip_path = None
        
        self.current_pdf_obj = None
        self.current_pdf_path = None
        self._pdf_pixmap_cache = collections.OrderedDict()
        self._pdf_cache_max = 32
        self._render_pool = QThreadPool.globalInstance()

        self.virtual_items = {}

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel(UI['labels_file_list'])
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

        self.btn_open = QPushButton("📁")
        self.btn_open.setToolTip(UI['buttons_open_folder_tooltip'])
        self.btn_open.setFixedSize(32, 28)
        self.btn_open.clicked.connect(self.select_directory)

        self.btn_help = QPushButton("❔")
        self.btn_help.setToolTip("UI['buttons_help_tooltip']")
        self.btn_help.setFixedSize(32, 28)
        self.btn_help.clicked.connect(self.open_help)

        self.btn_settings = QPushButton("⚙️")
        self.btn_settings.setFixedSize(32, 28)
        self.btn_settings.setToolTip(UI['buttons_settings_tooltip'])
        self.btn_settings.clicked.connect(self.open_settings)

        self.btn_fullscreen = QPushButton("⛶")
        self.btn_fullscreen.setToolTip(UI['buttons_fullscreen_tooltip'])
        self.btn_fullscreen.setFixedSize(32, 28)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        self.btn_close_all = QPushButton()
        self.btn_close_all.setToolTip(UI['buttons_close_all_tooltip'])
        self.btn_close_all.setText("🧹")
        self.btn_close_all.setFixedSize(32, 28)
        self.btn_close_all.clicked.connect(self.close_all_archives)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems([UI['scale_options_fit_page'], 
                                UI['scale_options_fit_height'],
                                UI['scale_options_fit_width']])
        self.scale_combo.currentIndexChanged.connect(self.on_scale_mode_changed)
        self.scale_combo.setMinimumWidth(65)

        left_top = QWidget()
        left_top_layout = QHBoxLayout(left_top)
        left_top_layout.setContentsMargins(2, 2, 2, 2)
        left_top_layout.addWidget(self.btn_open)
        left_top_layout.addWidget(self.btn_settings)
        left_top_layout.addWidget(self.btn_help)
        left_top_layout.addWidget(self.btn_fullscreen)
        left_top_layout.addWidget(self.btn_close_all)

        left_top_layout.addWidget(self.scale_combo)

        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        self.left_panel.setMinimumWidth(120)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(left_top)
        left_layout.addWidget(self.tree)

        self.image_label = ImageLabel()
        self.image_label.setText(UI['labels_image_placeholder'])
        self.image_label.setMinimumWidth(120)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.image_label)
        self.image_label.scroll_area = scroll
        scroll.setAlignment(Qt.AlignCenter)

        self.image_label.main_window = self

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self.scale_mode = "fit_page"

        self.custom_zoom = 100

        self.current_pixmap = None

        self._scroll_area = scroll
        self._viewport = self._scroll_area.viewport()

        default_percent = 0.12
        self.left_arrow = EdgeClickArea(self._viewport, direction="left", callback=self.prev_page, percent_width=default_percent)
        self.right_arrow = EdgeClickArea(self._viewport, direction="right", callback=self.next_page, percent_width=default_percent)

        self._viewport.installEventFilter(self)

        self.position_overlays()
        self.hide_overlays()

        self.load_config()
        auto_dir = self.config.get("general", {}).get("auto_open_dir", "")
        if auto_dir and Path(auto_dir).exists():
            self.current_dir = Path(auto_dir).resolve()
            self.load_directory()
            
        self.setup_shortcuts()

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_viewport_resized)

        try:
            vp = self._scroll_area.viewport()
            self._last_viewport_size = (vp.width(), vp.height())
        except Exception:
            self._last_viewport_size = (0, 0)

    def setup_shortcuts(self):
        try:
            old = getattr(self, "_shortcuts", None)
            if old:
                for sc in old:
                    try:
                        sc.disconnect()
                    except Exception:
                        pass
                    try:
                        sc.setParent(None)
                    except Exception:
                        pass
                self._shortcuts.clear()
        except Exception:
            pass

        self._shortcuts = []

        scfg = self.config.get("shortcuts", {})

        def mk(seq):
            if not seq:
                return QKeySequence()
            try:
                if isinstance(seq, QKeySequence):
                    return seq
                return QKeySequence(seq)
            except Exception:
                return QKeySequence()

        def add_shortcut(seq, slot, context=Qt.ApplicationShortcut):
            ks = mk(seq)
            sc = QShortcut(ks, self)
            sc.setContext(context)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)
            return sc

        add_shortcut(scfg.get("prev_page", "Left"), self.prev_page)
        add_shortcut(scfg.get("next_page", "Right"), self.next_page)
        add_shortcut(scfg.get("prev_archive", "Down"), self.prev_archive)
        add_shortcut(scfg.get("next_archive", "Up"), self.next_archive)
        add_shortcut(scfg.get("reset_zoom", "Ctrl+0"), lambda: self.set_scale_mode("fit_page"))
        add_shortcut(scfg.get("set_100", "Ctrl+1"), lambda: self.set_custom_zoom(100))

        add_shortcut("F11", self.toggle_fullscreen)


    def toggle_fullscreen(self):
        if self.isFullScreen():

            self.showNormal()

            try:
                self.btn_fullscreen.setText("⛶")
            except Exception:
                pass
        else:

            self.showFullScreen()
            try:
                self.btn_fullscreen.setText("🡽")
            except Exception:
                pass

        try:
            self.position_overlays()
            self.display_current_pixmap()
        except Exception:
            pass

    def toggle_file_list(self):
        try:
            visible = self.left_panel.isVisible()
            self.left_panel.setVisible(not visible)


            self.position_overlays()
            self.display_current_pixmap()
        except Exception:
            pass
        
    def _find_item_recursive(self, item: QTreeWidgetItem, target: str):
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        try:
            if data is not None and str(data) == target:
                return item
        except Exception:
            pass
        for i in range(item.childCount()):
            child = item.child(i)
            found = self._find_item_recursive(child, target)
            if found:
                return found
        return None

    def select_tree_item_for_path(self, image_path: str):
        if not image_path:
            return
        target = str(image_path)
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            found = self._find_item_recursive(top, target)
            if found:
                self.tree.setCurrentItem(found)
                try:
                    self.tree.scrollToItem(found)
                except Exception:
                    pass
                return

    def close_all_archives(self):
        try:

            if self.current_zip_obj is not None:
                try:
                    self.current_zip_obj.close()
                except Exception:
                    pass
                self.current_zip_obj = None
                self.current_zip_path = None
                
            if self.current_pdf_obj is not None:
                try:
                    self.current_pdf_obj.close()
                except Exception:
                    pass
                self.current_pdf_obj = None
                self.current_pdf_path = None

            for zip_path in list(self.virtual_items.keys()):
                virt = self.virtual_items.get(zip_path)
                if virt:
                    parent = virt.parent()
                    if parent is None:
                        idx = self.tree.indexOfTopLevelItem(virt)
                        if idx != -1:
                            self.tree.takeTopLevelItem(idx)
                    else:
                        parent.removeChild(virt)
            self.virtual_items.clear()


            self.image_list = []
            self.current_index = 0
            self.current_pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.hide_overlays()

            QMessageBox.information(self, UI["app_window_title"], UI['dialogs_info_close_archives'])

        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], f"{e}")



    def open_settings(self):
        dlg = SettingsDialog(self, config=self.config)
        if dlg.exec() == QDialog.Accepted:
            new_cfg = dlg.to_config()

            self.config.update(new_cfg)

            if "general" in new_cfg:
                self.config["general"].update(new_cfg["general"])
            if "shortcuts" in new_cfg:
                self.config["shortcuts"].update(new_cfg["shortcuts"])
            self.save_config()
            self.apply_settings()
            QMessageBox.information(self, UI['dialogs_info_saved'],
                                    UI['dialogs_info_saved_details'])


    def load_config(self):
        parser = configparser.ConfigParser()
        if CONFIG_PATH.exists():
            try:
                parser.read(CONFIG_PATH, encoding="utf-8")
                gen = dict(parser["general"]) if "general" in parser else {}
                sc = dict(parser["shortcuts"]) if "shortcuts" in parser else {}

                self.config["general"].update(gen)
                self.config["shortcuts"].update(sc)
            except Exception:

                pass
        else:

            try:
                parser["general"] = DEFAULT_CONFIG["general"]
                parser["shortcuts"] = DEFAULT_CONFIG["shortcuts"]
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    parser.write(f)
            except Exception:
                pass


        self.apply_settings()


    def save_config(self):
        parser = configparser.ConfigParser()
        parser["general"] = self.config.get("general", {})
        parser["shortcuts"] = self.config.get("shortcuts", {})
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                parser.write(f)
        except Exception as e:
            QMessageBox.warning(self, UI['dialogs_warning_save_failed'], str(e))

    def apply_settings(self):
        gen = self.config.get("general", {})
        sc = self.config.get("shortcuts", {})


        bg = gen.get("background_color", "#ffffff")
        try:

            self.image_label.setStyleSheet(f"background-color: {bg};")
            self._viewport.setStyleSheet(f"background-color: {bg};")
        except Exception:
            pass


        mode = gen.get("default_fit_mode", "fit_page")
        if mode in {"fit_page", "fit_height", "fit_width"}:
            self.scale_mode = mode

            if mode == "fit_page":
                self.scale_combo.setCurrentIndex(0)
            elif mode == "fit_height":
                self.scale_combo.setCurrentIndex(1)
            else:
                self.scale_combo.setCurrentIndex(2)


        try:
            self.setup_shortcuts()
        except Exception:
            pass

    def open_help(self):
        about_title = UI.get("dialogs_about_title", "About")
        help_html = UI.get("dialogs_help_html", None)
        QMessageBox.information(self, about_title, help_html)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, UI['buttons_open_folder_tooltip'])
        if dir_path:
            self.current_dir = Path(dir_path).resolve()
            self.load_directory()

    def load_directory(self):
        try:
            try:
                self.tree.setUpdatesEnabled(False)
                self.tree.blockSignals(True)
            except Exception:
                pass

            self.tree.clear()
            self.virtual_items.clear()
            items = []

            try:
                cur = (self.current_dir.resolve() if self.current_dir else Path.cwd().resolve())
            except Exception:
                cur = Path.cwd().resolve()

            parent = cur.parent if cur.parent != cur else None
            if parent and parent.exists():
                parent_item = QTreeWidgetItem(["../"])
                parent_item.setData(0, Qt.UserRole, f"dir://{str(parent.resolve())}")
                parent_item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
                items.append(parent_item)

            entries = []
            try:
                with os.scandir(cur) as it:
                    for de in it:
                        try:
                            if de.is_dir(follow_symlinks=False):
                                entries.append(('d', de))
                            elif de.is_file(follow_symlinks=False):
                                name_lower = de.name.lower()
                                if name_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) \
                                or name_lower.endswith(('.zip', '.cbz', '.pdf')):
                                    entries.append(('f', de))
                        except Exception:
                            continue
            except Exception:
                try:
                    for p in sorted(cur.iterdir(), key=self.sort_key):
                        try:
                            if p.is_dir():
                                entries.append(('d', p))
                            elif p.is_file():
                                ext = p.suffix.lower()
                                if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.zip', '.cbz', '.pdf'}:
                                    entries.append(('f', p))
                        except Exception:
                            continue
                except Exception:
                    self.current_dir = cur
                    return

            try:
                entries.sort(key=lambda t: self.sort_key(t[1]))
            except Exception:
                try:
                    entries.sort(key=lambda t: (t[1].name if hasattr(t[1], "name") else str(t[1])).lower())
                except Exception:
                    pass

            for typ, entry in entries:
                try:
                    if typ == 'd':
                        name = entry.name if hasattr(entry, "name") else Path(entry).name
                        try:
                            path_str = str(Path(entry.path if hasattr(entry, "path") else entry).resolve())
                        except Exception:
                            path_str = str(Path(entry.path if hasattr(entry, "path") else entry))

                        item = QTreeWidgetItem([name])
                        item.setData(0, Qt.UserRole, f"dir://{path_str}")
                        item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
                        items.append(item)
                    else:
                        name = entry.name if hasattr(entry, "name") else Path(entry).name
                        full_path = None
                        if hasattr(entry, "path"):
                            full_path = Path(entry.path)
                        else:
                            full_path = Path(entry)

                        item = QTreeWidgetItem([name])
                        item.setData(0, Qt.UserRole, str(full_path))
                        ext = full_path.suffix.lower()
                        if is_archive_ext(ext) or is_pdf_ext(ext):
                            item.setIcon(0, get_icon_for_file(full_path))
                        else:
                            item.setText(0, full_path.stem)
                        items.append(item)
                except Exception:
                    continue

            if items:
                try:
                    self.tree.insertTopLevelItems(0, items)
                except Exception:
                    for it in items:
                        try:
                            self.tree.addTopLevelItem(it)
                        except Exception:
                            pass

            self.current_dir = cur

        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], f"{e}")

        finally:
            try:
                self.tree.blockSignals(False)
                self.tree.setUpdatesEnabled(True)
            except Exception:
                pass


    def sort_key(self, path: Path):
        if self.sort_by_date:
            return path.stat().st_mtime
        return path.name.lower()
    
    def sort_key(self, path_or_entry):
        try:
            if self.sort_by_date:
                try:
                    if hasattr(path_or_entry, "stat") and hasattr(path_or_entry, "path") is False:
                        return path_or_entry.stat().st_mtime
                except Exception:
                    pass

                try:
                    if hasattr(path_or_entry, "path"):
                        return os.path.getmtime(path_or_entry.path)
                except Exception:
                    pass

                try:
                    p = Path(path_or_entry)
                    return p.stat().st_mtime
                except Exception:
                    name = getattr(path_or_entry, "name", None) or str(path_or_entry)
                    return natural_sort_key(name)

            else:
                name = None
                if hasattr(path_or_entry, "name"):
                    try:
                        name = path_or_entry.name
                    except Exception:
                        name = None
                if not name:
                    try:
                        name = Path(path_or_entry).name
                    except Exception:
                        name = str(path_or_entry)
                return natural_sort_key(name)

        except Exception:
            return natural_sort_key(str(path_or_entry))



    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return

        # dir
        if isinstance(data, str) and data.startswith("dir://"):
            try:
                target_dir = Path(str(data[6:])).resolve()
                if target_dir.exists() and target_dir.is_dir():
                    self.current_dir = target_dir
                    self.load_directory()
            except Exception:
                pass
            return

        # zip
        if isinstance(data, str) and data.startswith("zip://"):
            self.load_image(data)
            before_last, _ = data.rsplit(":", 1)
            zip_path = Path(before_last[6:]).resolve()
            virtual_item = self.virtual_items.get(str(zip_path))
            if virtual_item:
                self.image_list = [virtual_item.child(i).data(0, Qt.UserRole) for i in range(virtual_item.childCount())]
                try:
                    self.current_index = self.image_list.index(data)
                except ValueError:
                    self.current_index = 0
            return

        # pdf
        if isinstance(data, str) and data.startswith("pdf://"):
            self.load_image(data)
            before_last, _ = data.rsplit(":", 1)
            pdf_path = Path(before_last[6:]).resolve()
            virtual_item = self.virtual_items.get(str(pdf_path))
            if virtual_item:
                self.image_list = [virtual_item.child(i).data(0, Qt.UserRole) for i in range(virtual_item.childCount())]
                try:
                    self.current_index = self.image_list.index(data)
                except ValueError:
                    self.current_index = 0
            return

        # image file
        try:
            file_path = Path(str(data))
        except Exception:
            return

        ext = file_path.suffix.lower()
        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'} and file_path.exists():
            self.load_image(str(file_path))
            self.image_list = [str(file_path)]
            self.current_index = 0

            if self.current_zip_obj is not None:
                try:
                    self.current_zip_obj.close()
                except Exception:
                    pass
                self.current_zip_obj = None
                self.current_zip_path = None

            if self.current_pdf_obj is not None:
                try:
                    self.current_pdf_obj.close()
                except Exception:
                    pass
                self.current_pdf_obj = None
                self.current_pdf_path = None

    def on_item_double_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        if isinstance(data, str) and data.startswith("dir://"):
            try:
                target_dir = Path(str(data[6:])).resolve()
                if target_dir.exists() and target_dir.is_dir():
                    self.current_dir = target_dir
                    self.load_directory()
            except Exception:
                pass
            return

        if is_archive_path_str(data):
            zip_path = Path(data)
            self.extract_zip_to_tree(item, zip_path)
            
        elif is_pdf_path_str(data):
            pdf_path = Path(data)
            self.extract_pdf_to_tree(item, pdf_path)

    def extract_pdf_to_tree(self, parent_item, pdf_path: Path):
        try:
            pdf_path = Path(pdf_path).resolve()
            pdf_path_str = str(pdf_path)

            if pdf_path_str in self.virtual_items:
                virt = self.virtual_items[pdf_path_str]
                self.tree.setCurrentItem(virt)

                if self.current_pdf_obj is None or self.current_pdf_path != pdf_path_str:
                    if self.current_pdf_obj is not None:
                        try:
                            self.current_pdf_obj.close()
                        except Exception:
                            pass
                        self.current_pdf_obj = None
                        self.current_pdf_path = None

                    self.current_pdf_obj = fitz.open(pdf_path_str)
                    self.current_pdf_path = pdf_path_str

                pdf_files = [virt.child(i).data(0, Qt.UserRole) for i in range(virt.childCount())]
                self.image_list = pdf_files
                self.current_index = 0
                if pdf_files:
                    self.load_image(pdf_files[0])
                    self.tree.setCurrentItem(virt.child(0))
                return

            if self.current_pdf_obj is not None and self.current_pdf_path != pdf_path_str:
                try:
                    self.current_pdf_obj.close()
                except Exception:
                    pass
                self.current_pdf_obj = None
                self.current_pdf_path = None

            doc = fitz.open(pdf_path_str)

            virtual_item = QTreeWidgetItem([f"{UI['tree_expand_zip_prefix']}{pdf_path.name}"])
            virtual_item.setData(0, Qt.UserRole, pdf_path_str)
            virtual_item.setIcon(0, self.style().standardIcon(QStyle.SP_DirOpenIcon))

            pdf_files = []
            for p in range(len(doc)):
                label = f"Page {p+1}"
                sub_item = QTreeWidgetItem([label])
                sub_item.setData(0, Qt.UserRole, f"pdf://{pdf_path_str}:{p}")
                virtual_item.addChild(sub_item)
                pdf_files.append(f"pdf://{pdf_path_str}:{p}")

            if parent_item.parent() is None:
                index = self.tree.indexOfTopLevelItem(parent_item)
                self.tree.insertTopLevelItem(index + 1, virtual_item)
            else:
                parent = parent_item.parent()
                child_index = parent.indexOfChild(parent_item)
                parent.insertChild(child_index + 1, virtual_item)

            self.virtual_items[pdf_path_str] = virtual_item
            self.tree.expandItem(virtual_item)

            self.current_pdf_obj = doc
            self.current_pdf_path = pdf_path_str

            self.image_list = pdf_files
            self.current_index = 0
            if pdf_files:
                self.load_image(pdf_files[0])
                self.tree.setCurrentItem(virtual_item.child(0))

        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], f"{e}")

    def extract_zip_to_tree(self, parent_item, zip_path: Path):
        try:
            zip_path = Path(zip_path).resolve()
            zip_path_str = str(zip_path)

            if zip_path_str in self.virtual_items:
                virt = self.virtual_items[zip_path_str]
                self.tree.setCurrentItem(virt)

                if self.current_zip_obj is None or self.current_zip_path != zip_path_str:
                    if self.current_zip_obj is not None:
                        try:
                            self.current_zip_obj.close()
                        except Exception:
                            pass
                        self.current_zip_obj = None
                        self.current_zip_path = None

                    self.current_zip_obj = zipfile.ZipFile(zip_path_str, 'r')
                    self.current_zip_path = zip_path_str

                zip_files = [virt.child(i).data(0, Qt.UserRole) for i in range(virt.childCount())]
                self.image_list = zip_files
                self.current_index = 0
                if zip_files:
                    self.load_image(zip_files[0])
                    self.tree.setCurrentItem(virt.child(0))
                return

            if self.current_zip_obj is not None and self.current_zip_path != zip_path_str:
                try:
                    self.current_zip_obj.close()
                except Exception:
                    pass
                self.current_zip_obj = None
                self.current_zip_path = None

            zf = zipfile.ZipFile(zip_path_str, 'r')

            virtual_item = QTreeWidgetItem([f"{UI['tree_expand_zip_prefix']}{zip_path.name}"])
            virtual_item.setData(0, Qt.UserRole, zip_path_str)
            virtual_item.setIcon(0, self.style().standardIcon(QStyle.SP_DirOpenIcon))

            zip_files = []
            for name in zf.namelist():
                base = os.path.basename(name)
                if not base:
                    continue
                if base.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    sub_item = QTreeWidgetItem([base])
                    sub_item.setData(0, Qt.UserRole, f"zip://{zip_path_str}:{name}")
                    virtual_item.addChild(sub_item)
                    zip_files.append(f"zip://{zip_path_str}:{name}")

            if parent_item.parent() is None:
                index = self.tree.indexOfTopLevelItem(parent_item)
                self.tree.insertTopLevelItem(index + 1, virtual_item)
            else:
                parent = parent_item.parent()
                child_index = parent.indexOfChild(parent_item)
                parent.insertChild(child_index + 1, virtual_item)

            self.virtual_items[zip_path_str] = virtual_item
            self.tree.expandItem(virtual_item)

            self.current_zip_obj = zf
            self.current_zip_path = zip_path_str

            self.image_list = zip_files
            self.current_index = 0
            if zip_files:
                self.load_image(zip_files[0])
                self.tree.setCurrentItem(virtual_item.child(0))

        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], UI['dialogs_warning_zip_failed'].format(error=str(e)))
            
    def pre_render_adjacent_pages(self, pdf_path: str, page_num: int):
        try:
            pdf_path = str(Path(pdf_path).resolve())
        except Exception:
            return

        try:
            viewport = self._scroll_area.viewport()
            vw = max(1, viewport.width())
            vh = max(1, viewport.height())
        except Exception:
            vw, vh = 800, 600

        page_count = None
        try:
            if self.current_pdf_obj is not None and self.current_pdf_path == pdf_path:
                page_count = len(self.current_pdf_obj)
        except Exception:
            page_count = None

        if page_count is None:
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
            except Exception:
                page_count = None

        if page_count is None:
            return

        neighbors = []
        if page_num - 1 >= 0:
            neighbors.append(page_num - 1)
        if page_num + 1 < page_count:
            neighbors.append(page_num + 1)

        for p in neighbors:
            key = (pdf_path, p, int(vw), int(vh))
            if self._cache_get(key) is not None:
                continue
            try:
                self.request_pdf_page_render(pdf_path, p, vw, vh)
            except Exception:
                pass
            
    def clear_cache_and_rerender(self):
        if not self.image_list or not (0 <= self.current_index < len(self.image_list)):
            QMessageBox.information(self, UI["app_window_title"], UI.get('messages_no_selection', "No file selected"))
            return

        cur = self.image_list[self.current_index]
        try:
            cur_str = str(cur)
        except Exception:
            cur_str = ""

        if cur_str.startswith("pdf://"):
            try:
                before_last, p = cur_str.rsplit(":", 1)
                pdf_path = str(Path(before_last[6:]).resolve())
                page_num = int(p)
            except Exception:
                QMessageBox.warning(self, UI["app_window_title"], "Invalid PDF reference.")
                return

            try:
                self._pdf_pixmap_cache.clear()
            except Exception:
                try:
                    self._pdf_pixmap_cache = collections.OrderedDict()
                except Exception:
                    pass

            self.current_pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.hide_overlays()

            try:
                viewport = self._scroll_area.viewport()
                vw = max(1, viewport.width())
                vh = max(1, viewport.height())
            except Exception:
                vw, vh = 800, 600

            pm = None
            try:
                pm = self.request_pdf_page_render(pdf_path, page_num, vw, vh)
            except Exception:
                pm = None

            if pm:
                self.current_pixmap = pm
                self.display_current_pixmap()

            return

        if cur_str.startswith("zip://"):
            try:
                self.load_image(cur_str)
                QMessageBox.information(self, UI["app_window_title"], "Reloaded image from archive.")
            except Exception as e:
                QMessageBox.warning(self, UI["app_window_title"], f"Reload failed: {e}")
            return

        try:
            self.load_image(cur_str)
            QMessageBox.information(self, UI["app_window_title"], "Reloaded image file.")
        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], f"Reload failed: {e}")
            

    def load_image(self, image_path):
        try:
            if isinstance(image_path, str) and image_path.startswith("zip://"):
                # zip
                before_last, file_in_zip = image_path.rsplit(":", 1)
                zip_path = Path(before_last[6:]).resolve()
                zip_path_str = str(zip_path)

                if self.current_zip_obj is None or self.current_zip_path != zip_path_str:
                    if self.current_zip_obj is not None:
                        try:
                            self.current_zip_obj.close()
                        except Exception:
                            pass
                        self.current_zip_obj = None
                        self.current_zip_path = None
                    self.current_zip_obj = zipfile.ZipFile(zip_path_str, 'r')
                    self.current_zip_path = zip_path_str

                with self.current_zip_obj.open(file_in_zip) as f:
                    data = f.read()
                    pixmap = QPixmap()
                    pixmap.loadFromData(data)
                    self.current_pixmap = pixmap
                    self.display_current_pixmap()
                    try:
                        self.select_tree_item_for_path(image_path)
                    except Exception:
                        pass

            elif isinstance(image_path, str) and image_path.startswith("pdf://"):
                before_last, page_idx = image_path.rsplit(":", 1)
                pdf_path = str(Path(before_last[6:]).resolve())
                try:
                    page_num = int(page_idx)
                except Exception:
                    page_num = 0

                if self.current_pdf_obj is None or self.current_pdf_path != pdf_path:
                    try:
                        if self.current_pdf_obj is not None:
                            self.current_pdf_obj.close()
                    except Exception:
                        pass
                    try:
                        self.current_pdf_obj = fitz.open(pdf_path)
                        self.current_pdf_path = pdf_path
                    except Exception:
                        self.current_pdf_obj = None
                        self.current_pdf_path = None

                viewport = self._scroll_area.viewport()
                vw = max(1, viewport.width())
                vh = max(1, viewport.height())

                key = (pdf_path, page_num, vw, vh)
                cached = self._cache_get(key)
                if cached:
                    self.current_pixmap = cached
                    self.display_current_pixmap()
                    try:
                        self.select_tree_item_for_path(image_path)
                    except Exception:
                        pass
                    return

                pm = self.request_pdf_page_render(pdf_path, page_num, vw, vh)
                if pm:
                    self.current_pixmap = pm
                    self.display_current_pixmap()
                    try:
                        self.pre_render_adjacent_pages(pdf_path, page_num)
                    except Exception:
                        pass
                    try:
                        self.select_tree_item_for_path(image_path)
                    except Exception:
                        pass
                    return

            else:
                # image file
                pixmap = QPixmap(str(image_path))
                self.current_pixmap = pixmap
                self.display_current_pixmap()
                try:
                    self.select_tree_item_for_path(str(image_path))
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], f"{e}")

    def _cache_get(self, key):
        try:
            val = self._pdf_pixmap_cache.pop(key)
            self._pdf_pixmap_cache[key] = val
            return val
        except KeyError:
            return None

    def _cache_put(self, key, pixmap):
        if key in self._pdf_pixmap_cache:
            self._pdf_pixmap_cache.pop(key)
        self._pdf_pixmap_cache[key] = pixmap
        while len(self._pdf_pixmap_cache) > self._pdf_cache_max:
            self._pdf_pixmap_cache.popitem(last=False)
            
    def request_pdf_page_render(self, pdf_path: str, page_num: int, target_w: int, target_h: int):
        key = (pdf_path, page_num, int(target_w), int(target_h))
        cached = self._cache_get(key)
        if cached:
            return cached
        if not hasattr(self, "_render_signal"):
            self._render_signal = _RenderSignal()
            self._render_signal.finished.connect(self._on_pdf_render_finished)
        task = _PdfRenderTask(pdf_path, page_num, target_w, target_h, self._render_signal)
        self._render_pool.start(task)
        return None
    
    def _on_pdf_render_finished(self, pdf_path: str, page_num: int, pixmap: QPixmap):
        if pixmap and not pixmap.isNull():
            key = (pdf_path, page_num, pixmap.width(), pixmap.height())
            self._cache_put(key, pixmap)
        cur = None
        if self.image_list and 0 <= self.current_index < len(self.image_list):
            cur = self.image_list[self.current_index]
        if isinstance(cur, str) and cur.startswith("pdf://"):
            before_last, p = cur.rsplit(":", 1)
            try:
                cur_pdf = str(Path(before_last[6:]).resolve())
                cur_page = int(p)
            except Exception:
                cur_pdf, cur_page = None, None
            if cur_pdf == pdf_path and cur_page == page_num:
                pm = self._cache_get((pdf_path, page_num, pixmap.width(), pixmap.height()))
                if pm:
                    self.current_pixmap = pm
                    self.display_current_pixmap()


    def display_current_pixmap(self):
        if self.current_pixmap is None:
            self.image_label.setPixmap(QPixmap())
            self.hide_overlays()
            return

        scroll = self.image_label.scroll_area
        if not scroll:
            self.image_label.setPixmap(self.current_pixmap)
            return

        viewport = scroll.viewport()
        vw = max(1, viewport.width())
        vh = max(1, viewport.height())
        orig_w = self.current_pixmap.width()
        orig_h = self.current_pixmap.height()


        v_scroll_w = scroll.verticalScrollBar().sizeHint().width()
        h_scroll_h = scroll.horizontalScrollBar().sizeHint().height()


        def scaled_size_by_width(target_w):
            target_h = round(orig_h * (target_w / orig_w))
            return max(1, int(target_w)), max(1, int(target_h))

        def scaled_size_by_height(target_h):
            target_w = round(orig_w * (target_h / orig_h))
            return max(1, int(target_w)), max(1, int(target_h))


        if self.scale_mode == "fit_width":

            new_w, new_h = scaled_size_by_width(vw)

            if new_h > vh:
                avail_w = max(1, vw - v_scroll_w)
                new_w, new_h = scaled_size_by_width(avail_w)

        elif self.scale_mode == "fit_height":
            new_w, new_h = scaled_size_by_height(vh)

            if new_w > vw:
                avail_h = max(1, vh - h_scroll_h)
                new_w, new_h = scaled_size_by_height(avail_h)

        elif self.scale_mode == "custom":

            factor = max(0.01, self.custom_zoom / 100.0)
            new_w = max(1, round(orig_w * factor))
            new_h = max(1, round(orig_h * factor))


        else:  # fit_page

            ratio = min(vw / orig_w, vh / orig_h)
            new_w = max(1, round(orig_w * ratio))
            new_h = max(1, round(orig_h * ratio))



            if new_h > vh:
                avail_w = max(1, vw - v_scroll_w)
                ratio2 = min(avail_w / orig_w, vh / orig_h)
                new_w = max(1, round(orig_w * ratio2))
                new_h = max(1, round(orig_h * ratio2))

            if new_w > vw:
                avail_h = max(1, vh - h_scroll_h)
                ratio3 = min(vw / orig_w, avail_h / orig_h)
                new_w = max(1, round(orig_w * ratio3))
                new_h = max(1, round(orig_h * ratio3))


        scaled = self.current_pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)


        self.image_label.resize(scaled.size())


        allow_x = False
        allow_y = False
        if self.scale_mode == "fit_width":
            allow_x = False
            allow_y = scaled.height() > vh
        elif self.scale_mode == "fit_height":
            allow_y = False
            allow_x = scaled.width() > vw
        elif self.scale_mode == "custom":

            allow_x = scaled.width() > vw
            allow_y = scaled.height() > vh
        else:  # fit_page
            allow_x = scaled.width() > vw
            allow_y = scaled.height() > vh

        self.image_label.allow_pan_x = allow_x
        self.image_label.allow_pan_y = allow_y


        if self.image_label.drag_enabled and (allow_x or allow_y):
            self.image_label.setCursor(QCursor(Qt.OpenHandCursor))
        else:
            self.image_label.setCursor(QCursor(Qt.ArrowCursor))


        self.position_overlays()
        self.show_overlays()

        hbar = scroll.horizontalScrollBar()
        vbar = scroll.verticalScrollBar()
        if self.scale_mode == "fit_width":

            vbar.setValue(vbar.minimum())
        elif self.scale_mode == "fit_height":

            hbar.setValue(hbar.minimum())

    def on_scale_mode_changed(self, index):
        if index == 0:
            self.scale_mode = "fit_page"
        elif index == 1:
            self.scale_mode = "fit_height"
        else:
            self.scale_mode = "fit_width"
        self.display_current_pixmap()


    def prev_page(self):
        if self.image_list:
            self.current_index = (self.current_index - 1) % len(self.image_list)
            self.load_image(self.image_list[self.current_index])

    def next_page(self):
        if self.image_list:
            self.current_index = (self.current_index + 1) % len(self.image_list)
            self.load_image(self.image_list[self.current_index])

    def _ordered_expanded_virtuals(self):
        result = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            data = top.data(0, Qt.UserRole)
            if not data:
                continue
            try:
                data_str = str(data)
            except Exception:
                continue
            if is_archive_path_str(data_str):
                zip_path = str(Path(data_str).resolve())
                virt = self.virtual_items.get(zip_path)
                if virt:
                    result.append((zip_path, virt))
        return result


    def close_and_remove_current_virtual(self):
        try:
            if self.current_zip_obj is not None:
                try:
                    self.current_zip_obj.close()
                except Exception:
                    pass
                self.current_zip_obj = None
                self.current_zip_path = None
        except Exception:
            pass

    def _remove_virtual_item_by_path(self, zip_path_str):
        try:
            virt = self.virtual_items.get(zip_path_str)
            if not virt:
                return
            parent = virt.parent()
            if parent is None:
                idx = self.tree.indexOfTopLevelItem(virt)
                if idx != -1:
                    self.tree.takeTopLevelItem(idx)
            else:
                parent.removeChild(virt)
            try:
                del self.virtual_items[zip_path_str]
            except KeyError:
                pass
        except Exception:
            pass

    def _all_zip_top_level_items(self):
        out = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            data = top.data(0, Qt.UserRole)
            if not data:
                continue
            try:
                ds = str(data)
            except Exception:
                continue
            if is_archive_path_str(ds) or is_pdf_path_str(ds):
                zip_path = str(Path(ds).resolve())
                out.append((zip_path, top))
        return out

    def prev_archive(self):
        if not self.image_list or not (str(self.image_list[0]).startswith("zip://") or str(self.image_list[0]).startswith("pdf://")):
            return
        current_data = self.image_list[self.current_index]
        before_last, _ = current_data.rsplit(":", 1)
        current_zip_path = str(Path(before_last[6:]).resolve())
        ordered = self._all_zip_top_level_items()
        if not ordered:
            return
        zip_paths = [p for p, _ in ordered]
        try:
            curr_idx = zip_paths.index(current_zip_path)
        except ValueError:
            return
        prev_idx = (curr_idx - 1) % len(ordered)
        target_path_str, target_item = ordered[prev_idx]
        prev_path = current_zip_path

        try:
            if self.current_zip_obj is not None:
                try:
                    self.current_zip_obj.close()
                except Exception:
                    pass
                self.current_zip_obj = None
                self.current_zip_path = None
            if self.current_pdf_obj is not None:
                try:
                    self.current_pdf_obj.close()
                except Exception:
                    pass
                self.current_pdf_obj = None
                self.current_pdf_path = None
        except Exception:
            pass

        self._remove_virtual_item_by_path(prev_path)

        target_path = Path(target_path_str)
        if is_pdf_ext(target_path.suffix):
            self.extract_pdf_to_tree(target_item, target_path)
        else:
            self.extract_zip_to_tree(target_item, target_path)


    def next_archive(self):
        if not self.image_list or not (str(self.image_list[0]).startswith("zip://") or str(self.image_list[0]).startswith("pdf://")):
            return

        current_data = self.image_list[self.current_index]
        before_last, _ = current_data.rsplit(":", 1)
        current_zip_path = str(Path(before_last[6:]).resolve())
        ordered = self._all_zip_top_level_items()
        if not ordered:
            return
        zip_paths = [p for p, _ in ordered]
        try:
            curr_idx = zip_paths.index(current_zip_path)
        except ValueError:
            return
        next_idx = (curr_idx + 1) % len(ordered)
        target_path_str, target_item = ordered[next_idx]
        prev_path = current_zip_path

        try:
            if self.current_zip_obj is not None:
                try:
                    self.current_zip_obj.close()
                except Exception:
                    pass
                self.current_zip_obj = None
                self.current_zip_path = None
            if self.current_pdf_obj is not None:
                try:
                    self.current_pdf_obj.close()
                except Exception:
                    pass
                self.current_pdf_obj = None
                self.current_pdf_path = None
        except Exception:
            pass

        self._remove_virtual_item_by_path(prev_path)

        target_path = Path(target_path_str)
        if is_pdf_ext(target_path.suffix):
            self.extract_pdf_to_tree(target_item, target_path)
        else:
            self.extract_zip_to_tree(target_item, target_path)


    def switch_to_archive(self, virtual_item):
        children = [virtual_item.child(i).data(0, Qt.UserRole) for i in range(virtual_item.childCount())]
        if children:
            self.image_list = children
            self.current_index = 0
            self.load_image(children[0])
            self.tree.setCurrentItem(virtual_item.child(0))


    def show_context_menu(self, position):
        menu = QMenu()
        sort_action = QAction(UI['context_menu_sort_by_date'], self)
        sort_action.setCheckable(True)
        sort_action.setChecked(self.sort_by_date)
        sort_action.toggled.connect(self.toggle_sort)
        menu.addAction(sort_action)

        if self.image_list:
            prev_act = QAction(UI['context_menu_next'], self)
            prev_act.triggered.connect(self.prev_page)
            next_act = QAction(UI['context_menu_previous'], self)
            next_act.triggered.connect(self.next_page)
            menu.addSeparator()
            menu.addAction(prev_act)
            menu.addAction(next_act)
        
        menu.addSeparator()

        toggle_list_act = QAction(UI['context_menu_show_hide_file_panel'], self)
        toggle_list_act.triggered.connect(self.toggle_file_list)
        menu.addAction(toggle_list_act)

        menu.exec(self.tree.mapToGlobal(position))

    def toggle_sort(self):
        self.sort_by_date = not self.sort_by_date
        self.load_directory()


    def copy_current_to_clipboard(self):
        if not self.current_pixmap or self.current_pixmap.isNull():
            QMessageBox.information(self, UI["app_window_title"], UI['dialogs_info_clipboard_empty'])
            return
        clipboard = QApplication.clipboard()
        clipboard.setPixmap(self.current_pixmap)
        QMessageBox.information(self, UI["app_window_title"], UI['dialogs_info_copied'])

    def set_scale_mode(self, mode: str):
        if mode not in {"fit_page", "fit_width", "fit_height", "custom"}:
            return
        self.scale_mode = mode
        if mode == "fit_page":
            self.scale_combo.setCurrentIndex(0)
        elif mode == "fit_height":
            self.scale_combo.setCurrentIndex(1)
        elif mode == "fit_width":
            self.scale_combo.setCurrentIndex(2)
        self.display_current_pixmap()

    def set_custom_zoom(self, percent: int):
        try:
            percent = int(percent)
        except Exception:
            return
        self.custom_zoom = max(1, percent)
        self.scale_mode = "custom"
        self.display_current_pixmap()

    def reset_zoom(self):
        self.custom_zoom = 100
        self.scale_mode = "fit_page"
        self.scale_combo.setCurrentIndex(0)
        self.display_current_pixmap()

    def adjust_zoom_from_wheel(self, angle_delta_y: int):
        steps = angle_delta_y / 120.0
        change = int(round(steps * 10))
        new_zoom = self.custom_zoom + change
        new_zoom = max(1, min(1000, new_zoom))
        self.custom_zoom = new_zoom
        self.scale_mode = "custom"
        self.display_current_pixmap()


    def position_overlays(self):
        if not hasattr(self, "_viewport"):
            return
        vp = self._viewport
        vw = vp.width()
        vh = vp.height()


        left_w = self.left_arrow._computed_width(vw) if hasattr(self.left_arrow, "_computed_width") else self.left_arrow.width()
        right_w = self.right_arrow._computed_width(vw) if hasattr(self.right_arrow, "_computed_width") else self.right_arrow.width()


        self.left_arrow.setGeometry(QRect(0, 0, left_w, vh))
        self.right_arrow.setGeometry(QRect(max(0, vw - right_w), 0, right_w, vh))

        try:
            self.left_arrow.setFixedWidth(left_w)
        except Exception:
            pass
        try:
            self.right_arrow.setFixedWidth(right_w)
        except Exception:
            pass


    def show_overlays(self):
        if self.current_pixmap is None:
            return


        self.left_arrow.show()
        self.right_arrow.show()

    def hide_overlays(self):
        self.left_arrow.hide()
        self.right_arrow.hide()
    
    def eventFilter(self, watched, event):
        if watched is getattr(self, "_viewport", None):
            if event.type() == QEvent.Resize:
                self.position_overlays()
                try:
                    self._resize_timer.start(500)
                except Exception:
                    pass
                self.display_current_pixmap()
        return super().eventFilter(watched, event)
    
    def _on_viewport_resized(self):
        try:
            vp = self._scroll_area.viewport()
            vw, vh = max(1, vp.width()), max(1, vp.height())
        except Exception:
            return

        self._last_viewport_size = (vw, vh)

        if not self.image_list or not (0 <= self.current_index < len(self.image_list)):
            return

        cur = self.image_list[self.current_index]
        try:
            cur_str = str(cur)
        except Exception:
            cur_str = ""

        if cur_str.startswith("pdf://"):
            try:
                self.clear_cache_and_rerender()
            except Exception as e:
                # print(f"[warn] auto re-render on resize failed: {e}")
                pass

    def closeEvent(self, event):
        try:
            if self.current_zip_obj is not None:
                self.current_zip_obj.close()
                self.current_zip_obj = None
                self.current_zip_path = None
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ComicReader()
    window.show()
    sys.exit(app.exec())
