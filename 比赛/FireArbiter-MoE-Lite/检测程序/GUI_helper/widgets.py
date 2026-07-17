# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import *

class ClickableImageLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(str)

    def __init__(self, text: str = "", image_key: str = ""):
        super().__init__(text)
        self.image_key = str(image_key)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self.image_key)
        super().mousePressEvent(event)


class ImagePreviewDialog(QtWidgets.QDialog):
    def __init__(self, title: str, rgb: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 760)
        self.rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        self._temp_files: List[str] = []

        layout = QtWidgets.QVBoxLayout(self)
        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(720, 520)
        self.imageLabel.setStyleSheet("border:1px solid #CBD5E1; background:#0F172A;")
        layout.addWidget(self.imageLabel, 1)

        btn_layout = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("保存 PNG")
        open_btn = QtWidgets.QPushButton("用系统图片查看器打开")
        close_btn = QtWidgets.QPushButton("关闭")
        save_btn.clicked.connect(self.save_png)
        open_btn.clicked.connect(self.open_external_viewer)
        close_btn.clicked.connect(self.close)
        btn_layout.addStretch(1)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(open_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.update_pixmap()

    def update_pixmap(self):
        self.imageLabel.setPixmap(rgb_to_qpixmap(self.rgb, self.imageLabel, smooth=True))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def save_png(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存 attention 图片", "", "PNG Image (*.png)")
        if not path:
            return
        imwrite_png(path, self.rgb)

    def open_external_viewer(self):
        fd, path = tempfile.mkstemp(prefix="fire_attention_", suffix=".png")
        os.close(fd)
        imwrite_png(path, self.rgb)
        self._temp_files.append(path)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))


class BatchAuditWindow(QtWidgets.QDialog):
    def __init__(self, main_window: "MainWindow"):
        super().__init__(main_window)
        self.main_window = main_window
        self._updating = False
        self.setWindowTitle("批量检测审计")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.Window
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        self.resize(1280, 860)

        layout = QtWidgets.QVBoxLayout(self)
        self.topLabel = QtWidgets.QLabel("等待批量结果")
        self.topLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.topLabel.setMinimumHeight(58)
        self.topLabel.setStyleSheet("font-size:26px; font-weight:800; border-radius:8px; background:#EEF2F7; color:#334155;")
        layout.addWidget(self.topLabel)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.leftLabel = QtWidgets.QLabel("原图")
        self.rightLabel = QtWidgets.QLabel("mix attention")
        for label in [self.leftLabel, self.rightLabel]:
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumSize(520, 420)
            label.setStyleSheet("border:1px solid #CBD5E1; background:#0F172A; color:#E5E7EB;")
        splitter.addWidget(self.leftLabel)
        splitter.addWidget(self.rightLabel)
        splitter.setSizes([640, 640])
        layout.addWidget(splitter, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.indexLabel = QtWidgets.QLabel("0/0")
        self.indexLabel.setMinimumWidth(140)
        self.indexLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.filterCombo = QtWidgets.QComboBox()
        self.filterCombo.addItems(["全部", "有火", "无火", "错误"])
        self.filterCombo.setMinimumWidth(110)
        self.filterCombo.currentTextChanged.connect(lambda _=None: self.on_filter_changed())
        self.radioFire = QtWidgets.QRadioButton("有火")
        self.radioNoFire = QtWidgets.QRadioButton("无火")
        self.radioGroup = QtWidgets.QButtonGroup(self)
        self.radioGroup.addButton(self.radioFire, 1)
        self.radioGroup.addButton(self.radioNoFire, 0)
        self.prevBtn = QtWidgets.QPushButton("上一张 A")
        self.nextBtn = QtWidgets.QPushButton("下一张 D")
        self.exportWrongBtn = QtWidgets.QPushButton("导出错误图片")
        self.exportWrongBtn.setToolTip("批量导出模型判断错误的原图：优先使用人工审计标签；没有人工标签时使用已导入的对比 JSON。")
        self.saveBtn = QtWidgets.QPushButton("保存审计 JSON")
        self.prevBtn.clicked.connect(self.go_prev)
        self.nextBtn.clicked.connect(self.go_next)
        self.exportWrongBtn.clicked.connect(self.export_wrong_images)
        self.saveBtn.clicked.connect(self.export_audit_json)
        self.radioGroup.buttonClicked[int].connect(self.on_label_changed)
        for btn in [self.prevBtn, self.nextBtn, self.exportWrongBtn, self.saveBtn]:
            btn.setMinimumHeight(36)
        bottom.addWidget(self.indexLabel)
        bottom.addWidget(QtWidgets.QLabel("筛选："))
        bottom.addWidget(self.filterCombo)
        bottom.addStretch(1)
        bottom.addWidget(QtWidgets.QLabel("人工状态："))
        bottom.addWidget(self.radioFire)
        bottom.addWidget(self.radioNoFire)
        bottom.addStretch(1)
        bottom.addWidget(self.prevBtn)
        bottom.addWidget(self.nextBtn)
        bottom.addWidget(self.exportWrongBtn)
        bottom.addWidget(self.saveBtn)
        layout.addLayout(bottom)

        self.refresh_from_parent()

    def filtered_keys(self) -> List[str]:
        mw = self.main_window
        mode = self.filterCombo.currentText() if hasattr(self, "filterCombo") else "全部"
        keys: List[str] = []
        for key in mw.batch_order:
            item = mw.batch_results.get(key)
            if not item:
                continue
            final_label = mw.batch_final_label(item)
            keep = True
            if mode == "有火":
                keep = final_label == 1
            elif mode == "无火":
                keep = final_label == 0
            elif mode == "错误":
                truth = ground_truth_label_for_key(mw.batch_ground_truth or {}, key, item.get("path", ""))
                keep = truth is not None and int(truth) != int(final_label)
            if keep:
                keys.append(key)
        return keys

    def on_filter_changed(self):
        keys = self.filtered_keys()
        if keys:
            self.main_window.select_batch_key(keys[0])
        self.refresh_from_parent()

    def refresh_from_parent(self):
        mw = self.main_window
        keys = self.filtered_keys()
        key = mw.current_batch_key()
        if keys and key not in keys:
            mw.select_batch_key(keys[0])
            return
        total = len(keys)
        idx = (keys.index(key) + 1) if key in keys else 0
        self.indexLabel.setText(f"{idx}/{total}")
        if key is None or key not in mw.batch_results or key not in keys:
            mode = self.filterCombo.currentText() if hasattr(self, "filterCombo") else "全部"
            if mode == "错误" and not mw.batch_ground_truth:
                text = "请先导入对比 JSON，才能筛选错误项"
            else:
                text = f"当前筛选没有可审计结果：{mode}"
            self.topLabel.setText(text)
            self.topLabel.setStyleSheet("font-size:26px; font-weight:800; border-radius:8px; background:#EEF2F7; color:#334155;")
            self.leftLabel.clear()
            self.leftLabel.setText("原图")
            self.rightLabel.clear()
            self.rightLabel.setText("mix attention")
            return

        item = mw.batch_results[key]
        final_label = mw.batch_final_label(item)
        result = item.get("result", {})
        prob = result_prob(result, None)
        prob_text = "--" if prob is None else f"{prob:.4f}"
        state_text = "有火" if final_label == 1 else "无火"
        color = "#B91C1C" if final_label == 1 else "#064E3B"
        bg = "#FEE2E2" if final_label == 1 else "#DCFCE7"
        self.topLabel.setText(f"{state_text} | 判断概率：{prob_text} | {key}")
        self.topLabel.setStyleSheet(f"font-size:26px; font-weight:800; border-radius:8px; background:{bg}; color:{color};")

        try:
            raw_rgb = imread_rgb(item.get("path", ""))
            self.leftLabel.setPixmap(rgb_to_qpixmap(raw_rgb, self.leftLabel, smooth=True))
        except Exception as e:
            self.leftLabel.clear()
            self.leftLabel.setText(f"无法读取原图\n{e}")

        mix_rgb = mw.batch_mix_attention_rgb(item)
        if mix_rgb is not None:
            self.rightLabel.setPixmap(rgb_to_qpixmap(mix_rgb, self.rightLabel, smooth=True))
        else:
            self.rightLabel.clear()
            self.rightLabel.setText("没有 mix attention")

        self._updating = True
        try:
            self.radioFire.setChecked(final_label == 1)
            self.radioNoFire.setChecked(final_label == 0)
        finally:
            self._updating = False

    def on_label_changed(self, label: int):
        if self._updating:
            return
        self.main_window.set_batch_manual_label(int(label))
        self.refresh_from_parent()

    def go_prev(self):
        keys = self.filtered_keys()
        if not keys:
            self.refresh_from_parent()
            return
        current = self.main_window.current_batch_key()
        idx = keys.index(current) if current in keys else 0
        idx = max(0, idx - 1)
        self.main_window.select_batch_key(keys[idx])
        self.refresh_from_parent()

    def go_next(self):
        keys = self.filtered_keys()
        if not keys:
            self.refresh_from_parent()
            return
        current = self.main_window.current_batch_key()
        idx = keys.index(current) if current in keys else -1
        idx = min(len(keys) - 1, idx + 1)
        self.main_window.select_batch_key(keys[idx])
        self.refresh_from_parent()

    def export_audit_json(self):
        self.main_window.export_batch_audit_json()

    def export_wrong_images(self):
        self.main_window.export_batch_wrong_images()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_A:
            self.go_prev()
            return
        if key == QtCore.Qt.Key_D:
            self.go_next()
            return
        super().keyPressEvent(event)




class PromptSettingsDialog(QtWidgets.QDialog):
    def __init__(self, prompt_groups: Dict[str, List[str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("OWL-ViT Prompt 设置")
        self.resize(900, 720)
        self.editors: Dict[str, QtWidgets.QPlainTextEdit] = {}

        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "每行填写一个 prompt。正向、负向、疑似三组提示词共同生成空间语义图；"
            "保存后会立即重新编码提示词并重新推理。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        tabs = QtWidgets.QTabWidget()
        groups = prompt_groups_from_mapping(prompt_groups)
        for key, (label, _defaults) in PROMPT_GROUP_DEFS.items():
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)

            editor = QtWidgets.QPlainTextEdit()
            editor.setPlainText("\n".join(groups[key]))
            editor.setPlaceholderText("每行一个 prompt")
            editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            page_layout.addWidget(editor, 1)

            count_label = QtWidgets.QLabel(f"{len(groups[key])} 条")
            count_label.setAlignment(QtCore.Qt.AlignRight)
            page_layout.addWidget(count_label)
            editor.textChanged.connect(
                lambda editor=editor, count_label=count_label: count_label.setText(
                    f"{len(parse_prompt_text(editor.toPlainText()))} 条"
                )
            )

            tabs.addTab(page, label)
            self.editors[key] = editor

        layout.addWidget(tabs, 1)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        reset_btn = btn_box.addButton("恢复默认", QtWidgets.QDialogButtonBox.ResetRole)
        reset_btn.clicked.connect(self.reset_to_defaults)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def reset_to_defaults(self):
        defaults = default_prompt_groups()
        for key, editor in self.editors.items():
            editor.setPlainText("\n".join(defaults[key]))

    def prompt_groups(self) -> Dict[str, List[str]]:
        return prompt_groups_from_mapping({key: editor.toPlainText() for key, editor in self.editors.items()})


