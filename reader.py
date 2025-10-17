import collections
import configparser
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz
from PySide6.QtCore import (
    Qt, QPoint, QRect, QSize, QEvent, QLocale, QFileInfo, Signal, QObject, QRunnable, QThreadPool, QTimer
)
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut, QAction, QCursor, QPalette, QIcon, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMenu, QMessageBox,
    QStyle, QSplitter, QPushButton, QComboBox, QScrollArea, QFileIconProvider
)
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QColorDialog, QDialogButtonBox, QKeySequenceEdit, QLayout, QSizePolicy
)

from cvhelp import resize_qimage_with_opencv

APP_VERSION = "1.3-dev"


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
        },
        "thumbs_tooltip": {
            "zh": "打开缩略图面板",
            "en": "Open archive thumbnails",
            "ru": "Открыть миниатюры архива"
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
            "zh": "<h2>简单漫画阅读器</h2>\n"
                  f"<p><b>版本：</b> {APP_VERSION}</p>\n"
                  "<p><b>开发者：</b> Setsuna (github@puffdayo)</p>\n"
                  "<hr>\n"
                  "<p><b>使用说明：</b></p>\n"
                  "<ul>\n"
                  "<li><b>← / →</b>：上一页 / 下一页</li>\n"
                  "<li><b>↑ / ↓</b>：上一个 / 下一个压缩包</li>\n"
                  "<li><b>单击支持的文件</b>：展开或打开支持的文件</li>\n"
                  "<li><b>右键</b>：显示操作菜单</li>\n"
                  "<li><b>缩放模式：</b> 适应全页 / 适应高 / 适应宽 / 自定义百分比</li>\n"
                  "<li><b>滚轮：</b> 当图片超出窗口时平移</li>\n"
                  "<li><b>左右边缘点击：</b> 点击图片区域左右边缘可翻页</li>\n"
                  "<li><b>F11 或 ⛶ 按钮：</b> 切换全屏 / 退出全屏</li>\n"
                  "<li><b>隐藏文件面板：</b> 通过右键菜单或拖拽左右中间的分隔线至最左</li>\n"
                  "<li><b>显示文件面板：</b> 通过右键菜单或从最左边缘拖拽分隔线向右</li>\n"
                  "</ul>\n"
                  "<h4>支持的文件类型</h4>\n"
                  "<ul>\n"
                  "<li>图片：.jpg, .jpeg, .png, .gif, .bmp, .webp</li>\n"
                  "<li>压缩包：.zip, .cbz（单击可展开查看内部图片）</li>\n"
                  "<li>PDF 文档：.pdf（单击可展开逐页查看）</li>\n"
                  "</ul>\n"
                  "<hr>\n"
                  "<p>程序记忆设置保存到 <code>config.ini</code> 文件中。</p>",

            "en": "<h2>Simple Comic Reader</h2>\n"
                  f"<p><b>Version:</b> {APP_VERSION}</p>\n"
                  "<p><b>Developer:</b> Setsuna (github@puffdayo)</p>\n"
                  "<hr>\n"
                  "<p><b>Usage:</b></p>\n"
                  "<ul>\n"
                  "<li><b>← / →</b>: Previous / Next page</li>\n"
                  "<li><b>↑ / ↓</b>: Previous / Next archive</li>\n"
                  "<li><b>Click supported files</b>: Expand or open supported files</li>\n"
                  "<li><b>Right-click</b>: Show action menu</li>\n"
                  "<li><b>Scale modes:</b> Fit page / Fit height / Fit width / Custom %</li>\n"
                  "<li><b>Mouse wheel:</b> Pan when image exceeds window</li>\n"
                  "<li><b>Edge clicks:</b> Click left/right edges to flip pages</li>\n"
                  "<li><b>F11 or ⛶ button:</b> Toggle fullscreen</li>\n"
                  "<li><b>Hide file panel:</b> Use context menu or drag splitter left</li>\n"
                  "<li><b>Show file panel:</b> Drag splitter right from far left</li>\n"
                  "</ul>\n"
                  "<h4>Supported file types</h4>\n"
                  "<ul>\n"
                  "<li>Images: .jpg, .jpeg, .png, .gif, .bmp, .webp</li>\n"
                  "<li>Archives: .zip, .cbz (click to expand and view contained images)</li>\n"
                  "<li>PDF: .pdf (click to expand and view page-by-page)</li>\n"
                  "</ul>\n"
                  "<hr>\n"
                  "<p>Program stores settings in <code>config.ini</code>.</p>",

            "ru": "<h2>Простой просмотрщик комиксов</h2>\n"
                  f"<p><b>Версия:</b> {APP_VERSION}</p>\n"
                  "<p><b>Разработчик:</b> Setsuna (github@puffdayo)</p>\n"
                  "<hr>\n"
                  "<p><b>Инструкция:</b></p>\n"
                  "<ul>\n"
                  "<li><b>← / →</b>: Предыдущая / Следующая страница</li>\n"
                  "<li><b>↑ / ↓</b>: Предыдущий / Следующий архив</li>\n"
                  "<li><b>Клик по поддерживаемым файлам</b>: открыть или развернуть их</li>\n"
                  "<li><b>Правый клик</b>: Показать меню действий</li>\n"
                  "<li><b>Режимы масштабирования:</b> По странице / По высоте / По ширине / Процент</li>\n"
                  "<li><b>Колесо мыши:</b> Панорамирование, когда изображение больше окна</li>\n"
                  "<li><b>Клики по краям:</b> Нажатие слева/справа для перелистывания</li>\n"
                  "<li><b>F11 или ⛶:</b> Переключение полноэкранного режима</li>\n"
                  "<li><b>Скрыть панель файлов:</b> Через контекстное меню или переместить разделитель влево</li>\n"
                  "<li><b>Показать панель файлов:</b> Переместить разделитель вправо от самого левого края</li>\n"
                  "</ul>\n"
                  "<h4>Поддерживаемые типы файлов</h4>\n"
                  "<ul>\n"
                  "<li>Изображения: .jpg, .jpeg, .png, .gif, .bmp, .webp</li>\n"
                  "<li>Архивы: .zip, .cbz (клик — раскрыть и просмотреть файлы внутри)</li>\n"
                  "<li>PDF: .pdf (клик — просмотреть постранично)</li>\n"
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
        "thumbnails_title": {
            "zh": "缩略图",
            "en": "Thumbnails",
            "ru": "Миниатюры"
        },
        "thumbnails_no_images": {
            "zh": "未找到可用的缩略图。",
            "en": "No thumbnails found.",
            "ru": "Миниатюр не найдено."
        },
        "thumbnails_loading": {
            "zh": "正在生成缩略图…",
            "en": "Loading thumbnails…",
            "ru": "Загрузка миниатюр…"
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
        "refresh": {
            "zh": "刷新",
            "en": "Refresh",
            "ru": "Обновить"
        },
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
        },
        "thumbs_reload_failed": {
            "zh": "缩略图加载失败：{error}",
            "en": "Thumbnail load failed: {error}",
            "ru": "Ошибка загрузки миниатюр: {error}"
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
        color_row_layout.setContentsMargins(0, 0, 0, 0)
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
        # if (event.modifiers() & Qt.ControlModifier) and self.main_window:
        #     delta = event.angleDelta().y()
        #     if delta != 0:
        #         self.main_window.adjust_zoom_from_wheel(delta)
        #         event.accept()
        #         return
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

            # for pct in (50, 75, 100, 125, 150, 200):
            #     act = QAction(f"{pct}%", self)
            #     act.triggered.connect(lambda checked=False, p=pct: mw.set_custom_zoom(p))
            #     menu.addAction(act)
            #
            # menu.addSeparator()
            reset_act = QAction(UI['shortcuts_reset_zoom'], self)
            reset_act.triggered.connect(lambda: mw.set_scale_mode(f"{mw.scale_mode}"))
            menu.addAction(reset_act)

            self.demoire_act = QAction("Demoire: ON" if mw.demoire else "Demoire: OFF", self)
            self.demoire_act.triggered.connect(lambda: self._on_demoire_clicked(mw))
            menu.addAction(self.demoire_act)

            clear_cache_act = QAction(UI['context_menu_refresh'], self)
            clear_cache_act.triggered.connect(lambda: mw.clear_cache_and_rerender())
            menu.addAction(clear_cache_act)

            toggle_list_act = QAction(UI['context_menu_show_hide_file_panel'], self)
            toggle_list_act.triggered.connect(mw.toggle_file_list)
            menu.addAction(toggle_list_act)

        menu.exec(event.globalPos())

    def _on_demoire_clicked(self, mw):
        mw.trigger_demoire()
        self.demoire_act.setText("Demoire: ON" if mw.demoire else "Demoire: OFF")


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
    def __init__(self, pdf_path: str, page_num: int, sig: _RenderSignal,
                 target_w: int = None, target_h: int = None, max_scale: float = 8.0):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.sig = sig
        self.max_scale = float(max_scale)
        self.target_w = int(target_w) if target_w else None
        self.target_h = int(target_h) if target_h else None

    def run(self):
        try:
            doc = fitz.open(self.pdf_path)
            page = doc[self.page_num]
            rect = page.rect
            orig_w, orig_h = rect.width, rect.height

            scale = 1.0

            if self.target_w and self.target_h and orig_w and orig_h:
                try:
                    scale_w = float(self.target_w) / float(orig_w)
                    scale_h = float(self.target_h) / float(orig_h)
                    scale = max(1.0, min(scale_w, scale_h))
                except Exception:
                    scale = 1.0

            if (not self.target_w or not self.target_h) or scale <= 1.0:
                try:
                    imgs = page.get_images(full=True)
                    if imgs:
                        xref = imgs[0][0]
                        try:
                            imginfo = doc.extract_image(xref)
                            img_w = imginfo.get("width", 0)
                            if img_w and orig_w and orig_w > 0:
                                scale = max(scale, float(img_w) / float(orig_w))
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                if scale < 1.0:
                    scale = 1.0
            except Exception:
                scale = 1.0

            try:
                if scale > self.max_scale:
                    scale = self.max_scale
            except Exception:
                pass

            try:
                mat = fitz.Matrix(scale, scale)
            except Exception:
                mat = fitz.Matrix(1, 1)

            pix = page.get_pixmap(matrix=mat, alpha=False)

            n = pix.n
            if n == 3:
                fmt = QImage.Format_RGB888
            elif n == 4:
                fmt = QImage.Format_RGBA8888
            else:
                fmt = QImage.Format_RGB888

            try:
                qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
            except Exception:
                try:
                    png_bytes = pix.tobytes(output="png")
                except TypeError:
                    png_bytes = pix.tobytes("png")
                qimg = QImage.fromData(png_bytes)

            qpix = QPixmap.fromImage(qimg)

            try:
                self.sig.finished.emit(self.pdf_path, self.page_num, qpix)
            except Exception:
                pass

            try:
                doc.close()
            except Exception:
                pass

        except Exception:
            try:
                self.sig.finished.emit(self.pdf_path, self.page_num, QPixmap())
            except Exception:
                pass


class _ImageRenderTask(QRunnable):
    def __init__(self, zip_path_str: str, inner_name: str, target_w: int, target_h: int, sig):
        super().__init__()
        self.zip_path_str = zip_path_str
        self.inner_name = inner_name
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.sig = sig

    def run(self):
        try:
            with zipfile.ZipFile(self.zip_path_str, 'r') as zf:
                with zf.open(self.inner_name) as f:
                    data = f.read()
            qimg = QImage.fromData(data)
            if qimg is None or qimg.isNull():
                try:
                    self.sig.finished.emit(self.zip_path_str, self.inner_name, QImage())
                except Exception:
                    pass
                return

            try:
                max_dim = 8192
                if qimg.width() > max_dim or qimg.height() > max_dim:
                    qimg = qimg.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            except Exception:
                pass

            try:
                self.sig.finished.emit(self.zip_path_str, self.inner_name, qimg)
            except Exception:
                pass
        except Exception:
            try:
                self.sig.finished.emit(self.zip_path_str, self.inner_name, QImage())
            except Exception:
                pass


class _ImageRenderSignal(QObject):
    finished = Signal(str, str, QImage)  # zip_path_str, inner_name, QImage


class ThumbnailLabel(QLabel):
    clicked = Signal(object)

    def __init__(self, payload=None, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.payload)
        else:
            super().mousePressEvent(event)


class ThumbnailSignals(QObject):
    thumb_ready = Signal(object)
    finished = Signal(list)
    error = Signal(str)


class ThumbnailLoadTask(QRunnable):
    def __init__(self, dir_path: Path, thumb_w: int, thumb_h: int, signals: ThumbnailSignals, cancel_cb=None):
        super().__init__()
        self.dir_path = Path(dir_path) if dir_path else None
        self.thumb_w = int(thumb_w)
        self.thumb_h = int(thumb_h)
        self.signals = signals
        self.cancel_cb = cancel_cb or (lambda: False)

    def run(self):
        if not self.dir_path:
            try:
                self.signals.finished.emit([])
            except Exception:
                pass
            return

        results = []
        try:
            entries = []
            for p in sorted(self.dir_path.iterdir(), key=lambda x: natural_sort_key(x.name)):
                if p.is_file():
                    ext = p.suffix.lower()
                    if ext in {'.zip', '.cbz', '.pdf'} or ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}:
                        entries.append(p)
        except Exception as e:
            try:
                self.signals.error.emit(str(e))
            except Exception:
                pass
            try:
                self.signals.finished.emit(results)
            except Exception:
                pass
            return

        for p in entries:
            if self.cancel_cb():
                break

            try:
                ext = p.suffix.lower()
                key = str(p.resolve())
                caption = p.name

                if ext in {'.zip', '.cbz'}:
                    try:
                        with zipfile.ZipFile(str(p), 'r') as zf:
                            names = [n for n in zf.namelist() if
                                     os.path.basename(n) and os.path.basename(n).lower().endswith(
                                         ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
                            if not names:
                                continue
                            names.sort(key=natural_sort_key)
                            first = names[0]
                            with zf.open(first) as f:
                                data = f.read()
                            thumb_key = f"zip://{str(p.resolve())}:{first}"
                            payload = {'key': thumb_key, 'data': data, 'caption': caption}
                            results.append(payload)
                            try:
                                self.signals.thumb_ready.emit(payload)
                            except Exception:
                                pass
                    except Exception:
                        continue

                elif ext == '.pdf':
                    try:
                        doc = fitz.open(str(p))
                        if len(doc) <= 0:
                            doc.close()
                            continue
                        page = doc[0]
                        scale = max(0.05, min(1.0, self.thumb_w / page.rect.width))
                        mat = fitz.Matrix(scale, scale)
                        pixmap = page.get_pixmap(matrix=mat, alpha=False)
                        try:
                            png_bytes = pixmap.tobytes(output="png")
                        except TypeError:
                            png_bytes = pixmap.tobytes("png")
                        doc.close()
                        thumb_key = f"pdf://{str(p.resolve())}:0"
                        payload = {'key': thumb_key, 'data': png_bytes, 'caption': caption}
                        results.append(payload)
                        try:
                            self.signals.thumb_ready.emit(payload)
                        except Exception:
                            pass
                    except Exception:
                        try:
                            doc.close()
                        except Exception:
                            pass
                        continue

                else:
                    try:
                        with open(p, 'rb') as f:
                            data = f.read()
                        thumb_key = str(p)
                        payload = {'key': thumb_key, 'data': data, 'caption': caption}
                        results.append(payload)
                        try:
                            self.signals.thumb_ready.emit(payload)
                        except Exception:
                            pass
                    except Exception:
                        continue

            except Exception:
                continue

        try:
            self.signals.finished.emit(results)
        except Exception:
            pass


class ThumbnailDialog(QDialog):
    def __init__(self, parent, thumb_size=(240, 160), spacing=8):
        super().__init__(parent)
        self.setWindowTitle(UI['dialogs_thumbnails_title'])
        self.parent = parent
        self.thumb_w, self.thumb_h = thumb_size
        self.spacing = spacing
        self.setModal(False)
        self.resize(768, 600)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.flow = FlowLayout(self.container, margin=6, spacing=self.spacing)
        self.container.setLayout(self.flow)
        self.scroll.setWidget(self.container)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll)
        self.setLayout(layout)

        self._thumb_cache = {}  # key->QPixmap
        self._items = []  # (thumb_label, key)
        self._thumb_task_cancelled = False
        self._thumb_task = None
        self._thumb_signals = None

    def clear_thumbs(self):
        self._thumb_task_cancelled = True
        self._thumb_task = None
        while self.flow.count():
            it = self.flow.takeAt(0)
            wid = it.widget()
            if wid:
                wid.setParent(None)
        self._thumb_cache.clear()
        self._items.clear()

    def _on_thumb_ready(self, payload: Dict[str, Any]):
        try:
            data = payload.get('data') if isinstance(payload, dict) else None
            key = payload.get('key') if isinstance(payload, dict) else None
            caption = payload.get('caption', "")
            if not data or not key:
                return
            pix = QPixmap()
            ok = pix.loadFromData(data)
            if not ok or pix.isNull():
                return
            self.add_thumbnail_widget(key, pix, caption)
        except Exception:
            pass

    def _on_thumbs_finished(self, results: List[Dict[str, Any]]):
        self._thumb_task = None

    def add_thumbnail_widget(self, key, pixmap: QPixmap, caption: str = ""):
        thumb = pixmap.scaled(self.thumb_w, self.thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl = ThumbnailLabel(payload=key, parent=self.container)
        lbl.setFixedSize(self.thumb_w, self.thumb_h + 24)
        frame = QWidget()
        v = QVBoxLayout(frame)
        v.setContentsMargins(4, 4, 4, 4)
        image_lbl = QLabel()
        image_lbl.setAlignment(Qt.AlignCenter)
        image_lbl.setPixmap(thumb)
        image_lbl.setFixedSize(self.thumb_w, self.thumb_h)
        cap = QLabel(caption)
        cap.setAlignment(Qt.AlignCenter)
        cap.setWordWrap(True)
        cap.setFixedHeight(24)
        v.addWidget(image_lbl)
        v.addWidget(cap)
        frame.setFixedSize(self.thumb_w + 8, self.thumb_h + 26)
        click_wrapper = ThumbnailLabel(payload=key, parent=self.container)
        inner_layout = QVBoxLayout(click_wrapper)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(frame)
        click_wrapper.setFixedSize(frame.size())
        click_wrapper.clicked.connect(self.on_thumb_clicked)
        self.flow.addWidget(click_wrapper)
        self._items.append((click_wrapper, key))

    def on_thumb_clicked(self, key):
        try:
            if not isinstance(key, str):
                return

            if key.startswith("zip://"):
                try:
                    before_last, inner = key.rsplit(":", 1)
                    zip_path_str = before_last[6:]
                    zip_path = Path(zip_path_str).resolve()
                except Exception:
                    self.parent.load_image(key)
                    return

                found_item = None
                for i in range(self.parent.tree.topLevelItemCount()):
                    top = self.parent.tree.topLevelItem(i)
                    data = top.data(0, Qt.UserRole)
                    try:
                        if data and str(data) == str(zip_path):
                            found_item = top
                            break
                        if isinstance(data, str) and data == str(zip_path):
                            found_item = top
                            break
                    except Exception:
                        continue

                if not found_item:
                    for i in range(self.parent.tree.topLevelItemCount()):
                        top = self.parent.tree.topLevelItem(i)

                        def _rec_search(item):
                            nonlocal found_item
                            if found_item:
                                return
                            d = item.data(0, Qt.UserRole)
                            try:
                                if d and str(d) == str(zip_path):
                                    found_item = item
                                    return
                            except Exception:
                                pass
                            for j in range(item.childCount()):
                                _rec_search(item.child(j))

                        _rec_search(top)
                        if found_item:
                            break

                if found_item is not None:
                    try:
                        self.parent.extract_zip_to_tree(found_item, zip_path)
                    except Exception:
                        pass

                virt = self.parent.virtual_items.get(str(zip_path))
                target_ref = f"zip://{str(zip_path)}:{inner}"
                if virt:
                    files = [virt.child(i).data(0, Qt.UserRole) for i in range(virt.childCount())]
                    self.parent.image_list = files
                    try:
                        self.parent.current_index = files.index(target_ref)
                    except ValueError:
                        found_idx = None
                        for idx, it in enumerate(files):
                            if it.endswith(inner):
                                found_idx = idx
                                break
                        if found_idx is not None:
                            self.parent.current_index = found_idx
                            target_ref = files[found_idx]
                        else:
                            self.parent.current_index = 0
                            target_ref = files[0] if files else target_ref

                    try:
                        self.parent.load_image(target_ref)
                        try:
                            self.parent.select_tree_item_for_path(target_ref)
                        except Exception:
                            pass
                        return
                    except Exception:
                        pass

                try:
                    self.parent.load_image(target_ref)
                except Exception:
                    try:
                        self.parent.load_image(key)
                    except Exception:
                        pass
                return

            if key.startswith("pdf://"):
                try:
                    before_last, p = key.rsplit(":", 1)
                    pdf_path_str = before_last[6:]
                    pdf_path = Path(pdf_path_str).resolve()
                    page_idx = int(p)
                except Exception:
                    self.parent.load_image(key)
                    return

                found_item = None
                for i in range(self.parent.tree.topLevelItemCount()):
                    top = self.parent.tree.topLevelItem(i)
                    data = top.data(0, Qt.UserRole)
                    try:
                        if data and str(data) == str(pdf_path):
                            found_item = top
                            break
                    except Exception:
                        continue

                if found_item is not None:
                    try:
                        self.parent.extract_pdf_to_tree(found_item, pdf_path)
                    except Exception:
                        pass

                virt = self.parent.virtual_items.get(str(pdf_path))
                target_ref = f"pdf://{str(pdf_path)}:{page_idx}"
                if virt:
                    files = [virt.child(i).data(0, Qt.UserRole) for i in range(virt.childCount())]
                    self.parent.image_list = files
                    try:
                        self.parent.current_index = files.index(target_ref)
                    except ValueError:
                        self.parent.current_index = max(0, min(len(files) - 1, page_idx))
                        target_ref = files[self.parent.current_index] if files else target_ref

                    try:
                        self.parent.load_image(target_ref)
                        try:
                            self.parent.select_tree_item_for_path(target_ref)
                        except Exception:
                            pass
                        return
                    except Exception:
                        pass

                try:
                    self.parent.load_image(target_ref)
                except Exception:
                    try:
                        self.parent.load_image(key)
                    except Exception:
                        pass
                return

            try:
                self.parent.image_list = [key]
                self.parent.current_index = 0
                self.parent.load_image(key)
                try:
                    self.parent.select_tree_item_for_path(key)
                except Exception:
                    pass
            except Exception:
                try:
                    self.parent.load_image(key)
                except Exception:
                    pass

        except Exception:
            pass

    def populate_from_dir(self, dir_path: Path):
        self.clear_thumbs()
        self._thumb_task_cancelled = False

        if dir_path is None:
            QMessageBox.information(self, UI['dialogs_thumbnails_title'],
                                    UI['dialogs_thumbnails_no_images'])
            return

        signals = ThumbnailSignals()
        signals.thumb_ready.connect(self._on_thumb_ready)
        signals.finished.connect(self._on_thumbs_finished)
        signals.error.connect(lambda e: None)  # TODO: show error msg

        self._thumb_signals = signals

        def cancel_cb():
            return self._thumb_task_cancelled or (not self.isVisible())

        task = ThumbnailLoadTask(dir_path=dir_path, thumb_w=self.thumb_w, thumb_h=self.thumb_h, signals=signals,
                                 cancel_cb=cancel_cb)
        self._thumb_task = task

        QThreadPool.globalInstance().start(task)


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self.itemList = []

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        mleft, mtop, mright, mbottom = self.getContentsMargins()
        size += QSize(mleft + mright, mtop + mbottom)
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        effective_rect = rect.adjusted(+self.contentsMargins().left(), +self.contentsMargins().top(),
                                       -self.contentsMargins().right(), -self.contentsMargins().bottom())
        x = effective_rect.x()
        y = effective_rect.y()
        maxWidth = effective_rect.width()

        for item in self.itemList:
            wid = item.widget()
            hint = item.sizeHint()
            nextX = x + hint.width() + self._spacing
            if nextX - self._spacing > effective_rect.right() + 1 and lineHeight > 0:
                x = effective_rect.x()
                y = y + lineHeight + self._spacing
                nextX = x + hint.width() + self._spacing
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), hint))

            x = nextX
            lineHeight = max(lineHeight, hint.height())

        return y + lineHeight - rect.y() + self.contentsMargins().bottom()


class ComicReader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI["app_window_title"])
        self.setWindowIcon(QIcon('icon-512.png'))
        self.setGeometry(100, 40, 1650, 1010)

        self.config = {"general": dict(DEFAULT_CONFIG["general"]), "shortcuts": dict(DEFAULT_CONFIG["shortcuts"])}
        self.current_dir = Path()
        self.sort_by_date = False
        self.image_list = []
        self.current_index = 0
        self._loading_dir = False

        self.current_zip_obj = None
        self.current_zip_path = None

        self.current_pdf_obj = None
        self.current_pdf_path = None
        self._pdf_pixmap_cache = collections.OrderedDict()
        self._pdf_cache_max = 128
        self._render_pool = QThreadPool.globalInstance()
        self._render_pool.setMaxThreadCount(4)

        self._zip_img_cache = collections.OrderedDict()
        self._zip_img_cache_max = 128
        self._zip_pending_requests = {}
        self._image_render_pool = QThreadPool.globalInstance()
        self._image_render_pool.setMaxThreadCount(4)
        self._image_render_signal = _ImageRenderSignal()
        self._image_render_signal.finished.connect(self._on_image_render_finished)
        self.demoire = False

        self.virtual_items = {}

        self._thumbnail_dialog = None

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel(UI['labels_file_list'])
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

        self.btn_open = QPushButton("📁")
        self.btn_open.setToolTip(UI['buttons_open_folder_tooltip'])
        self.btn_open.setFixedSize(32, 28)
        self.btn_open.clicked.connect(self.select_directory)

        self.btn_thumbs = QPushButton("🖼️")
        self.btn_thumbs.setToolTip(UI['buttons_thumbs_tooltip'])
        self.btn_thumbs.setFixedSize(32, 28)
        self.btn_thumbs.clicked.connect(self.open_archive_thumbnails)

        self.btn_help = QPushButton("❔")
        self.btn_help.setToolTip(UI['buttons_help_tooltip'])
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
        left_top_layout.addWidget(self.btn_thumbs)

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
        self.left_arrow = EdgeClickArea(self._viewport, direction="left", callback=self.prev_page,
                                        percent_width=default_percent)
        self.right_arrow = EdgeClickArea(self._viewport, direction="right", callback=self.next_page,
                                         percent_width=default_percent)

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

        self._debug_zip_cache = False

    def _dbg(self, *args):
        try:
            if getattr(self, "_debug_zip_cache", False):
                print("[ComicReader DEBUG]", *args)
        except Exception:
            pass

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
            if getattr(self, "_thumbnail_dialog", None):
                try:
                    self._thumbnail_dialog.close()
                except Exception:
                    pass
                self._thumbnail_dialog = None

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

            # tree
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

            # cache
            try:
                if hasattr(self, "_pdf_pixmap_cache"):
                    self._pdf_pixmap_cache.clear()
            except Exception:
                try:
                    self._pdf_pixmap_cache = collections.OrderedDict()
                except Exception:
                    pass

            try:
                if hasattr(self, "_zip_img_cache"):
                    self._zip_img_cache.clear()
            except Exception:
                try:
                    self._zip_img_cache = collections.OrderedDict()
                except Exception:
                    pass

            self.image_list = []
            self.current_index = 0
            self.current_pixmap = None
            try:
                self.image_label.setPixmap(QPixmap())
            except Exception:
                pass
            self.hide_overlays()

        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], f"{e}")

    def open_archive_thumbnails(self):
        try:
            if self._thumbnail_dialog is None:
                self._thumbnail_dialog = ThumbnailDialog(self, thumb_size=(160, 240), spacing=8)
            curdir = self.current_dir if (self.current_dir and self.current_dir.exists()) else None
            if curdir:
                self._thumbnail_dialog.populate_from_dir(curdir)
            self._thumbnail_dialog.show()
            self._thumbnail_dialog.raise_()
            self._thumbnail_dialog.activateWindow()
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
        if getattr(self, "_loading_dir", False):
            return

        dir_path = QFileDialog.getExistingDirectory(self, UI['buttons_open_folder_tooltip'])
        if dir_path:
            self.current_dir = Path(dir_path).resolve()
            self.load_directory()

    def reload_directory(self):
        self.close_all_archives()
        self.load_directory()

    def load_directory(self):
        try:
            if self._thumbnail_dialog is not None:
                try:
                    self._thumbnail_dialog.close()
                except Exception:
                    pass
                self._thumbnail_dialog = None
        except Exception:
            pass

        if getattr(self, "_loading_dir", False):
            return

        self._loading_dir = True

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
                                if name_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')) \
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
                                if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.zip', '.cbz', '.pdf'}:
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
                            item.setIcon(0, get_icon_for_file(full_path))
                            item.setText(0, name)
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
            self._loading_dir = False

    # def sort_key(self, path: Path):
    #     if self.sort_by_date:
    #         return path.stat().st_mtime
    #     return path.name.lower()

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
        if getattr(self, "_loading_dir", False):
            return

        data = item.data(0, Qt.UserRole)
        if data is None:
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

        try:
            if isinstance(data, str) and is_archive_path_str(data):
                zip_path = Path(data)
                self.extract_zip_to_tree(item, zip_path)
                return
        except Exception:
            pass

        try:
            if isinstance(data, str) and is_pdf_path_str(data):
                pdf_path = Path(data)
                self.extract_pdf_to_tree(item, pdf_path)
                return
        except Exception:
            pass

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

        try:
            file_path = Path(str(data))
        except Exception:
            return

        ext = file_path.suffix.lower()
        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'} and file_path.exists():
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
                label = f"Page {p + 1}"
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
                if base.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
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

        # pdf
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
            # self.image_label.setPixmap(QPixmap())
            # self.hide_overlays()

            try:
                viewport = self._scroll_area.viewport()
                vw = max(1, viewport.width())
                vh = max(1, viewport.height())
            except Exception:
                vw, vh = 800, 600

            try:
                self.request_pdf_page_render(pdf_path, page_num, vw, vh)
            except Exception:
                pass

            try:
                self.pre_render_adjacent_pages(pdf_path, page_num)
            except Exception:
                pass

            return

        # zip
        if cur_str.startswith("zip://"):
            try:
                before_last, inner = cur_str.rsplit(":", 1)
                zip_path = Path(before_last[6:]).resolve()
                zip_path_str = str(zip_path)
            except Exception:
                QMessageBox.warning(self, UI["app_window_title"], "Invalid ZIP reference.")
                return

            try:
                keys = list(self._zip_img_cache.keys())
                for k in keys:
                    if isinstance(k, tuple) and len(k) >= 1 and str(k[0]) == zip_path_str:
                        try:
                            self._zip_img_cache.pop(k)
                        except Exception:
                            pass
            except Exception:
                pass

            self.current_pixmap = None
            # self.image_label.setPixmap(QPixmap())
            # self.hide_overlays()

            try:
                viewport = self._scroll_area.viewport()
                vw = max(1, viewport.width())
                vh = max(1, viewport.height())
            except Exception:
                vw, vh = 800, 600

            try:
                self.request_zip_image_render(zip_path_str, inner, vw, vh)
            except Exception:
                pass

            try:
                self._prefetch_zip_neighbors(zip_path_str, inner, vw, vh)
            except Exception:
                pass

            return

        # image file
        try:
            self.load_image(cur_str)

        except Exception as e:
            QMessageBox.warning(self, UI["app_window_title"], f"Reload failed: {e}")

    def load_image(self, image_path):
        try:
            if isinstance(image_path, str) and image_path.startswith("zip://"):
                before_last, file_in_zip = image_path.rsplit(":", 1)
                zip_path = Path(before_last[6:]).resolve()
                zip_path_str = str(zip_path)

                viewport = self._scroll_area.viewport()
                vw = max(1, viewport.width())
                vh = max(1, viewport.height())

                key = (zip_path_str, file_in_zip, vw, vh)
                cached_qimg = self._zip_cache_get(key)
                if cached_qimg:
                    try:
                        self.current_pixmap = QPixmap.fromImage(cached_qimg)
                        self.display_current_pixmap()
                        try:
                            self.select_tree_item_for_path(image_path)
                        except Exception:
                            pass
                        self._prefetch_zip_neighbors(zip_path_str, file_in_zip, vw, vh)
                        return
                    except Exception:
                        pass

                try:
                    self.request_zip_image_render(zip_path_str, file_in_zip, vw, vh)
                except Exception:
                    pass

                try:
                    self._prefetch_zip_neighbors(zip_path_str, file_in_zip, vw, vh)
                except Exception:
                    pass
                return


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

    def _prefetch_zip_neighbors(self, zip_path_str: str, inner_name: str, vw: int, vh: int):
        try:
            virt = self.virtual_items.get(zip_path_str)
            if not virt:
                return
            files = [virt.child(i).data(0, Qt.UserRole) for i in range(virt.childCount())]
            target_ref = f"zip://{zip_path_str}:{inner_name}"
            try:
                idx = files.index(target_ref)
            except ValueError:
                idx = None
                for i, it in enumerate(files):
                    if it.endswith(inner_name):
                        idx = i
                        break
                if idx is None:
                    return
            neighbors = []
            if idx - 1 >= 0:
                neighbors.append(files[idx - 1])
            if idx + 1 < len(files):
                neighbors.append(files[idx + 1])
            for nref in neighbors:
                try:
                    _, inner = nref.rsplit(":", 1)
                    self.request_zip_image_render(zip_path_str, inner, vw, vh)
                except Exception:
                    pass
        except Exception:
            pass

    def _to_physical_size(self, logical_w: int, logical_h: int, dpr: float) -> Tuple[int, int]:
        try:
            pw = max(1, int(round(logical_w * (dpr or 1.0))))
            ph = max(1, int(round(logical_h * (dpr or 1.0))))
            return pw, ph
        except Exception:
            return max(1, logical_w), max(1, logical_h)

    def _pixmap_physical_size(self, pm: QPixmap) -> Tuple[int, int]:
        try:
            try:
                dpr = float(pm.devicePixelRatioF())
            except Exception:
                dpr = float(pm.devicePixelRatio()) if hasattr(pm, "devicePixelRatio") else 1.0
            w = max(1, int(round(pm.width() * dpr)))
            h = max(1, int(round(pm.height() * dpr)))
            return w, h
        except Exception:
            return max(1, pm.width()), max(1, pm.height())

    def _qimage_physical_size(self, qimg: QImage) -> Tuple[int, int]:
        try:
            return max(1, qimg.width()), max(1, qimg.height())
        except Exception:
            return 1, 1

    def _cache_get(self, key):
        try:
            pdf_path, page_num, tw, th = key
        except Exception:
            return None

        candidates = []
        try:
            for k in list(self._pdf_pixmap_cache.keys()):
                try:
                    k_pdf, k_page, k_w, k_h = k
                except Exception:
                    continue
                if k_pdf == pdf_path and k_page == page_num and k_w > 0 and k_h > 0:
                    if k_w >= tw and k_h >= th:
                        candidates.append((k_w * k_h, k))
        except Exception:
            pass

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1][2]))
            chosen_key = candidates[0][1]
            try:
                val = self._pdf_pixmap_cache.pop(chosen_key)
                self._pdf_pixmap_cache[chosen_key] = val
                return val
            except Exception:
                return None

        return None

    def _cache_put(self, key, pixmap):
        try:
            if key in self._pdf_pixmap_cache:
                self._pdf_pixmap_cache.pop(key)
            self._pdf_pixmap_cache[key] = pixmap
            while len(self._pdf_pixmap_cache) > self._pdf_cache_max:
                self._pdf_pixmap_cache.popitem(last=False)
        except Exception:
            try:
                self._pdf_pixmap_cache[key] = pixmap
            except Exception:
                pass

    def _zip_cache_put(self, key, qimage):
        try:
            if key in self._zip_img_cache:
                self._zip_img_cache.pop(key)
            self._zip_img_cache[key] = qimage
            while len(self._zip_img_cache) > self._zip_img_cache_max:
                self._zip_img_cache.popitem(last=False)
        except Exception:
            try:
                self._zip_img_cache[key] = qimage
            except Exception:
                pass

    def _zip_cache_get(self, key):
        try:
            zip_path, inner, tw, th = key
        except Exception:
            return None

        # exact hit
        try:
            val = self._zip_img_cache.get(key)
            if val is not None:
                try:
                    self._zip_img_cache.pop(key)
                    self._zip_img_cache[key] = val
                except Exception:
                    pass
                try:
                    if val.width() >= tw and val.height() >= th:
                        return val
                except Exception:
                    return val
        except Exception:
            pass

        candidates = []
        try:
            for k in list(self._zip_img_cache.keys()):
                try:
                    k_zip, k_inner, k_w, k_h = k
                except Exception:
                    continue
                if k_zip == zip_path and k_inner == inner and k_w > 0 and k_h > 0:
                    if k_w >= tw and k_h >= th:
                        candidates.append((k_w * k_h, k))
        except Exception:
            pass

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1][2]))
            chosen = candidates[0][1]
            try:
                val = self._zip_img_cache.pop(chosen)
                self._zip_img_cache[chosen] = val
                try:
                    c_w, c_h = chosen[2], chosen[3]
                    if c_w >= tw and c_h >= th:
                        scaled_qimg = val.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        new_key = (zip_path, inner, int(scaled_qimg.width()), int(scaled_qimg.height()))
                        try:
                            self._zip_img_cache[new_key] = scaled_qimg
                            while len(self._zip_img_cache) > self._zip_img_cache_max:
                                self._zip_img_cache.popitem(last=False)
                        except Exception:
                            pass
                        if scaled_qimg.width() >= tw and scaled_qimg.height() >= th:
                            return scaled_qimg
                        else:
                            return val
                except Exception:
                    pass
                return val
            except Exception:
                pass

        return None

    def trigger_demoire(self):
        self.demoire = not self.demoire

        try:
            if not self.image_list or self.current_index < 0 or self.current_index >= len(self.image_list):
                QMessageBox.information(self, "Info", "No image loaded.")
                return

            cur_item = self.image_list[self.current_index]
            if not cur_item.startswith("zip://"):
                QMessageBox.information(self, "Info", "Demoire only works for ZIP images.")
                return

            if hasattr(self, "_zip_img_cache"):
                self._zip_img_cache.clear()
            if hasattr(self, "_pdf_pixmap_cache"):
                self._pdf_pixmap_cache.clear()

            self.load_image(cur_item)

        except Exception as e:
            QMessageBox.warning(self, "Demoire Error", str(e))

    def request_zip_image_render(self, zip_path_str: str, inner_name: str, target_w: int, target_h: int):
        try:
            viewport = self._scroll_area.viewport()
            try:
                dpr = float(viewport.devicePixelRatioF())
            except Exception:
                try:
                    dpr = float(self.devicePixelRatioF())
                except Exception:
                    dpr = 1.0
        except Exception:
            dpr = 1.0

        tw_px, th_px = self._to_physical_size(int(target_w), int(target_h), dpr)
        key = (zip_path_str, inner_name, int(tw_px), int(th_px))

        self._dbg("request_zip_image_render target logical:", target_w, target_h, "dpr:", dpr, "=> phys:", tw_px, th_px,
                  "key:", key)

        cached = self._zip_cache_get(key)
        if cached:
            self._dbg("cache HIT exact/>= target for", key, "->", (cached.width(), cached.height()))
            return cached

        pk = (zip_path_str, inner_name)
        try:
            s = self._zip_pending_requests.get(pk)
            if s is None:
                s = set()
                self._zip_pending_requests[pk] = s
            s.add((int(tw_px), int(th_px), str(self.scale_mode)))
        except Exception:
            try:
                self._zip_pending_requests[pk] = {(int(tw_px), int(th_px), str(self.scale_mode))}
            except Exception:
                pass

        self._dbg("cache MISS for", key, "-> registered pending:", self._zip_pending_requests.get(pk))

        task = _ImageRenderTask(zip_path_str, inner_name, int(tw_px), int(th_px), self._image_render_signal)
        self._image_render_pool.start(task)
        return None

    def _do_fit_scale(self, tw, th, qimage: QImage, mode: str):
        if mode == "fit_width":
            scaled = qimage.scaledToWidth(int(tw), Qt.SmoothTransformation)
        elif mode == "fit_height":
            scaled = qimage.scaledToHeight(int(th), Qt.SmoothTransformation)
        else:
            scaled = qimage.scaled(int(tw), int(th), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return scaled

    def _on_image_render_finished(self, zip_path_str: str, inner_name: str, qimage: QImage):
        try:
            if qimage is None or qimage.isNull():
                return

            phys_w = int(qimage.width())
            phys_h = int(qimage.height())

            orig_key_exact = (zip_path_str, inner_name, phys_w, phys_h)
            try:
                self._zip_cache_put(orig_key_exact, qimage)
            except Exception:
                try:
                    self._zip_img_cache[orig_key_exact] = qimage
                except Exception:
                    pass

            orig_key = (zip_path_str, inner_name, -1, -1)
            try:
                existing_orig = self._zip_img_cache.get(orig_key)
                replace_orig = False
                if existing_orig is None:
                    replace_orig = True
                else:
                    try:
                        ex_w, ex_h = int(existing_orig.width()), int(existing_orig.height())
                        if phys_w > ex_w or phys_h > ex_h:
                            replace_orig = True
                    except Exception:
                        replace_orig = True
                if replace_orig:
                    try:
                        if orig_key in self._zip_img_cache:
                            self._zip_img_cache.pop(orig_key)
                        self._zip_img_cache[orig_key] = qimage
                        while len(self._zip_img_cache) > self._zip_img_cache_max:
                            self._zip_img_cache.popitem(last=False)
                    except Exception:
                        try:
                            self._zip_img_cache[orig_key] = qimage
                        except Exception:
                            pass
            except Exception:
                pass

            pk = (zip_path_str, inner_name)
            pending = set()
            try:
                pending = self._zip_pending_requests.pop(pk, set())
            except Exception:
                pending = set()

            for (tw, th, mode) in pending:
                try:
                    if phys_w >= 1 and phys_h >= 1:
                        if self.demoire:
                            scaled = resize_qimage_with_opencv(qimage, int(tw), int(th), mode=mode,
                                                               use_comic_rescale=True)
                        else:
                            scaled = self._do_fit_scale(tw, th.th, qimage, mode)

                        scaled_key = (zip_path_str, inner_name, int(scaled.width()), int(scaled.height()))
                        try:
                            self._zip_cache_put(scaled_key, scaled)
                            self._dbg("generated scaled cache for", scaled_key, "from orig", (phys_w, phys_h), "mode:",
                                      mode)
                        except Exception:
                            try:
                                self._zip_img_cache[scaled_key] = scaled
                            except Exception:
                                pass
                except Exception:
                    continue

            cur = None
            if self.image_list and 0 <= self.current_index < len(self.image_list):
                cur = self.image_list[self.current_index]
            if isinstance(cur, str) and cur.startswith("zip://"):
                try:
                    before_last, cur_inner = cur.rsplit(":", 1)
                    cur_zip = str(Path(before_last[6:]).resolve())
                except Exception:
                    cur_zip = None
                    cur_inner = None

                if cur_zip == zip_path_str and cur_inner == inner_name:
                    try:
                        viewport = self._scroll_area.viewport()
                        try:
                            dpr = float(viewport.devicePixelRatioF())
                        except Exception:
                            try:
                                dpr = float(self.devicePixelRatioF())
                            except Exception:
                                dpr = 1.0
                        vw = max(1, viewport.width())
                        vh = max(1, viewport.height())
                        tw_px, th_px = self._to_physical_size(vw, vh, dpr)
                    except Exception:
                        tw_px, th_px = phys_w, phys_h

                    chosen_qimg = None
                    try:
                        mode = str(self.scale_mode)
                        if self.demoire:
                            chosen_qimg = resize_qimage_with_opencv(qimage, int(tw_px), int(th_px), mode=mode,
                                                                    use_comic_rescale=True)
                            print('2')
                        else:
                            chosen_qimg = self._do_fit_scale(tw_px, th_px, qimage, mode)

                        try:
                            self._zip_cache_put(
                                (zip_path_str, inner_name, int(chosen_qimg.width()), int(chosen_qimg.height())),
                                chosen_qimg)
                        except Exception:
                            pass
                    except Exception:
                        chosen_qimg = qimage

                    try:
                        self._dbg("DEBUG scale_mode:", mode,
                                  "requested phys target (tw_px,th_px):", tw_px, th_px,
                                  "orig qimage size:", qimage.width(), qimage.height(),
                                  "chosen_qimg size:", chosen_qimg.width() if chosen_qimg else None,
                                  chosen_qimg.height() if chosen_qimg else None,
                                  "viewport dpr:", dpr)
                    except Exception:
                        pass

                    if chosen_qimg is not None:
                        try:
                            pix = QPixmap.fromImage(chosen_qimg)
                            try:
                                pix.setDevicePixelRatioF(dpr)
                            except Exception:
                                try:
                                    pix.setDevicePixelRatio(dpr)
                                except Exception:
                                    pass
                            self.current_pixmap = pix
                            self.display_current_pixmap()
                            self._dbg("displayed CHOSEN QIMAGE -> pixsize:", pix.width(), pix.height(), "dpr set:", dpr)
                        except Exception:
                            # fallback
                            try:
                                pix = QPixmap.fromImage(qimage)
                                try:
                                    pix.setDevicePixelRatioF(dpr)
                                except Exception:
                                    pass
                                self.current_pixmap = pix
                                self.display_current_pixmap()
                            except Exception:
                                pass
        except Exception:
            pass

    def request_pdf_page_render(self, pdf_path: str, page_num: int, target_w: int, target_h: int):
        try:
            viewport = self._scroll_area.viewport()
            try:
                dpr = float(viewport.devicePixelRatioF())
            except Exception:
                try:
                    dpr = float(self.devicePixelRatioF())
                except Exception:
                    dpr = 1.0
        except Exception:
            dpr = 1.0

        vw_px, vh_px = self._to_physical_size(int(target_w), int(target_h), dpr)
        key = (pdf_path, page_num, int(vw_px), int(vh_px))

        cached = self._cache_get(key)
        if cached:
            return cached

        if not hasattr(self, "_render_signal"):
            self._render_signal = _RenderSignal()
            self._render_signal.finished.connect(self._on_pdf_render_finished)

        task = _PdfRenderTask(pdf_path, page_num, self._render_signal, target_w=int(vw_px), target_h=int(vh_px))
        self._render_pool.start(task)
        return None

    def _on_pdf_render_finished(self, pdf_path: str, page_num: int, pixmap: QPixmap):
        try:
            if not pixmap or pixmap.isNull():
                return
            phys_w, phys_h = self._pixmap_physical_size(pixmap)

            key = (pdf_path, page_num, phys_w, phys_h)
            try:
                self._cache_put(key, pixmap)
            except Exception:
                try:
                    if key in self._pdf_pixmap_cache:
                        self._pdf_pixmap_cache.pop(key)
                    self._pdf_pixmap_cache[key] = pixmap
                except Exception:
                    pass

            orig_key = (pdf_path, page_num, -1, -1)
            try:
                existing = self._pdf_pixmap_cache.get(orig_key)
                replace = False
                if existing is None:
                    replace = True
                else:
                    try:
                        ex_w, ex_h = self._pixmap_physical_size(existing)
                        if phys_w > ex_w or phys_h > ex_h:
                            replace = True
                    except Exception:
                        replace = True
                if replace:
                    try:
                        if orig_key in self._pdf_pixmap_cache:
                            self._pdf_pixmap_cache.pop(orig_key)
                        self._pdf_pixmap_cache[orig_key] = pixmap
                        while len(self._pdf_pixmap_cache) > self._pdf_cache_max:
                            self._pdf_pixmap_cache.popitem(last=False)
                    except Exception:
                        try:
                            self._pdf_pixmap_cache[orig_key] = pixmap
                        except Exception:
                            pass
            except Exception:
                pass

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
                    try:
                        try:
                            dpr = float(self._scroll_area.viewport().devicePixelRatioF())
                        except Exception:
                            try:
                                dpr = float(self.devicePixelRatioF())
                            except Exception:
                                dpr = 1.0
                        try:
                            pixmap.setDevicePixelRatioF(dpr)
                        except Exception:
                            try:
                                pixmap.setDevicePixelRatio(dpr)
                            except Exception:
                                pass

                        self.current_pixmap = pixmap
                        self.image_label.setPixmap(pixmap)
                        logical_w = int(round(pixmap.width() / (dpr or 1.0)))
                        logical_h = int(round(pixmap.height() / (dpr or 1.0)))
                        self.image_label.resize(logical_w, logical_h)
                    except Exception:
                        try:
                            self.display_current_pixmap()
                        except Exception:
                            pass
        except Exception:
            pass

    def display_current_pixmap(self):
        if self.current_pixmap is None:
            # self.image_label.setPixmap(QPixmap())
            # self.hide_overlays()
            return

        scroll = self.image_label.scroll_area
        if not scroll:
            self.image_label.setPixmap(self.current_pixmap)
            return

        viewport = scroll.viewport()
        vw = max(1, viewport.width())
        vh = max(1, viewport.height())

        try:
            viewport_dpr = float(viewport.devicePixelRatioF())
        except Exception:
            try:
                viewport_dpr = float(self.devicePixelRatioF())
            except Exception:
                viewport_dpr = 1.0

        vw_px = max(1, int(round(vw * viewport_dpr)))
        vh_px = max(1, int(round(vh * viewport_dpr)))

        pm = self.current_pixmap

        try:
            try:
                pm_dpr = float(pm.devicePixelRatioF())
            except Exception:
                try:
                    pm_dpr = float(pm.devicePixelRatio())
                except Exception:
                    pm_dpr = 1.0
            orig_w = max(1, int(round(pm.width() * pm_dpr)))
            orig_h = max(1, int(round(pm.height() * pm_dpr)))
        except Exception:
            orig_w = max(1, pm.width())
            orig_h = max(1, pm.height())
            pm_dpr = 1.0

        v_scroll_w = scroll.verticalScrollBar().sizeHint().width()
        h_scroll_h = scroll.horizontalScrollBar().sizeHint().height()

        def scaled_size_by_width_px(target_w_px):
            target_h_px = max(1, int(round(orig_h * (float(target_w_px) / max(1, orig_w)))))
            return max(1, int(target_w_px)), target_h_px

        def scaled_size_by_height_px(target_h_px):
            target_w_px = max(1, int(round(orig_w * (float(target_h_px) / max(1, orig_h)))))
            return target_w_px, max(1, int(target_h_px))

        if self.scale_mode == "fit_width":
            new_w_px, new_h_px = scaled_size_by_width_px(vw_px)
            if new_h_px > vh_px:
                avail_w = max(1, (vw - v_scroll_w) * viewport_dpr)
                new_w_px, new_h_px = scaled_size_by_width_px(int(round(avail_w)))

        elif self.scale_mode == "fit_height":
            new_w_px, new_h_px = scaled_size_by_height_px(vh_px)
            if new_w_px > vw_px:
                avail_h = max(1, (vh - h_scroll_h) * viewport_dpr)
                new_w_px, new_h_px = scaled_size_by_height_px(int(round(avail_h)))

        elif self.scale_mode == "custom":
            factor = max(0.01, self.custom_zoom / 100.0)
            new_w_px = max(1, int(round(orig_w * factor)))
            new_h_px = max(1, int(round(orig_h * factor)))

        else:  # fit_page
            ratio = min(vw_px / max(1, orig_w), vh_px / max(1, orig_h))
            new_w_px = max(1, int(round(orig_w * ratio)))
            new_h_px = max(1, int(round(orig_h * ratio)))

            if new_h_px > vh_px:
                avail_w_px = max(1, (vw - v_scroll_w) * viewport_dpr)
                ratio2 = min(avail_w_px / max(1, orig_w), vh_px / max(1, orig_h))
                new_w_px = max(1, int(round(orig_w * ratio2)))
                new_h_px = max(1, int(round(orig_h * ratio2)))

            if new_w_px > vw_px:
                avail_h_px = max(1, (vh - h_scroll_h) * viewport_dpr)
                ratio3 = min(vw_px / max(1, orig_w), avail_h_px / max(1, orig_h))
                new_w_px = max(1, int(round(orig_w * ratio3)))
                new_h_px = max(1, int(round(orig_h * ratio3)))

        try:
            if self.demoire:
                qimg = pm.toImage()
                scaled_qimg = resize_qimage_with_opencv(qimg, int(new_w_px), int(new_h_px), mode="keep_aspect",
                                                        use_comic_rescale=True)
                scaled = QPixmap.fromImage(scaled_qimg)
                print('3')
            else:
                scaled = pm.scaled(int(new_w_px), int(new_h_px), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception:
            scaled = pm

        try:
            scaled.setDevicePixelRatioF(viewport_dpr)
        except Exception:
            try:
                scaled.setDevicePixelRatio(viewport_dpr)
            except Exception:
                pass

        try:
            logical_w = int(round(scaled.width() / viewport_dpr))
            logical_h = int(round(scaled.height() / viewport_dpr))
            self.image_label.setPixmap(scaled)
            self.image_label.resize(logical_w, logical_h)
        except Exception:
            try:
                self.image_label.setPixmap(scaled)
            except Exception:
                pass

        allow_x = False
        allow_y = False
        if self.scale_mode == "fit_width":
            allow_x = False
            allow_y = scaled.height() / viewport_dpr > vh
        elif self.scale_mode == "fit_height":
            allow_y = False
            allow_x = scaled.width() / viewport_dpr > vw
        elif self.scale_mode == "custom":
            allow_x = scaled.width() / viewport_dpr > vw
            allow_y = scaled.height() / viewport_dpr > vh
        else:  # fit_page
            allow_x = scaled.width() / viewport_dpr > vw
            allow_y = scaled.height() / viewport_dpr > vh

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
        if not self.image_list or not (
                str(self.image_list[0]).startswith("zip://") or str(self.image_list[0]).startswith("pdf://")):
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
        if not self.image_list or not (
                str(self.image_list[0]).startswith("zip://") or str(self.image_list[0]).startswith("pdf://")):
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

        reload_act = QAction(UI['context_menu_refresh'], self)
        reload_act.triggered.connect(self.reload_directory)
        menu.addAction(reload_act)

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
        try:
            if mode == "fit_page":
                self.scale_combo.setCurrentIndex(0)
            elif mode == "fit_height":
                self.scale_combo.setCurrentIndex(1)
            elif mode == "fit_width":
                self.scale_combo.setCurrentIndex(2)
        except Exception:
            pass

        try:
            if hasattr(self, "_pdf_pixmap_cache"):
                try:
                    self._pdf_pixmap_cache.clear()
                except Exception:
                    try:
                        import collections
                        self._pdf_pixmap_cache = collections.OrderedDict()
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if hasattr(self, "_pdf_rendering"):
                try:
                    self._pdf_rendering.clear()
                except Exception:
                    try:
                        self._pdf_rendering = set()
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            cur = None
            if self.image_list and 0 <= self.current_index < len(self.image_list):
                cur = self.image_list[self.current_index]
            cur_str = str(cur) if cur else ""
            if cur_str.startswith("pdf://"):
                try:
                    self.clear_cache_and_rerender()
                except Exception:
                    try:
                        self.display_current_pixmap()
                    except Exception:
                        pass
            else:
                try:
                    self.display_current_pixmap()
                except Exception:
                    pass
        except Exception:
            pass

    def set_custom_zoom(self, percent: int):
        try:
            percent = int(percent)
        except Exception:
            return
        self.custom_zoom = max(1, percent)
        self.scale_mode = "custom"

        try:
            cur = self.image_list[self.current_index] if (
                    self.image_list and 0 <= self.current_index < len(self.image_list)) else None
            cur_str = str(cur) if cur else ""
            if cur_str.startswith("pdf://") or cur_str.startswith("zip://"):
                self.clear_cache_and_rerender()
                return
        except Exception:
            pass

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

        left_w = self.left_arrow._computed_width(vw) if hasattr(self.left_arrow,
                                                                "_computed_width") else self.left_arrow.width()
        right_w = self.right_arrow._computed_width(vw) if hasattr(self.right_arrow,
                                                                  "_computed_width") else self.right_arrow.width()

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
                    self._resize_timer.start(250)
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

        if (vw, vh) == getattr(self, "_last_viewport_size", (0, 0)):
            return
        lw, lh = getattr(self, "_last_viewport_size", (0, 0))
        if abs(vw - lw) <= 2 and abs(vh - lh) <= 2:
            return

        self._last_viewport_size = (vw, vh)
        if not self.image_list or not (0 <= self.current_index < len(self.image_list)):
            return
        cur = self.image_list[self.current_index]
        try:
            cur_str = str(cur)
        except Exception:
            cur_str = ""
        if cur_str.startswith("pdf://") or cur_str.startswith("zip://"):
            try:
                self.clear_cache_and_rerender()
            except Exception:
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
    window.showMaximized()
    sys.exit(app.exec())
