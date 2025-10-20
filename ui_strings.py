APP_VERSION = "0.3.2-alpha"

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
        },
        "close_all_archives": {
            "zh": "关闭所有归档",
            "en": "Close all archives",
            "ru": "Закрыть все архивы"
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
                  "<li><b>右键菜单还可以打开高质量的 De-moire 图像渲染算法</b></li>\n"
                  "<li><b>数字键 1 / 2 / 3</b>：切换到 适应全页 / 适应宽 / 适应高（默认）</li>\n"
                  "<li><b>数字键 5</b>：关闭所有已打开的压缩包</li>\n"
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
                  "<li><b>Inside right-click menu there's a high-quality de-moire image rendering option,</b></li>\n"
                  "<li><b>Number keys 1 / 2 / 3</b>: Switch to Fit page / Fit width / Fit height (defaults)</li>\n"
                  "<li><b>Number key 5</b>: Close all opened archives</li>\n"
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
                  "<li><b>В контекстном меню доступен фильтр de-moire для повышения качества изображения</b></li>\n"
                  "<li><b>Цифровые клавиши 1 / 2 / 3</b>: Переключиться на По странице / По ширине / По высоте (по умолчанию)</li>\n"
                  "<li><b>Клавиша 5</b>: Закрыть все открытые архивы</li>\n"
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
