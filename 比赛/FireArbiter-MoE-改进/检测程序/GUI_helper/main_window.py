# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import *
from .engine import FireInferenceEngine
from .workers import InferenceRunnable, BatchInferenceRunnable, FrameReaderThread, ModelLoadThread
from .logging_utils import logger
from .widgets import ClickableImageLabel, ImagePreviewDialog, BatchAuditWindow, PromptSettingsDialog

class MainWindow(QtWidgets.QMainWindow):
    SOURCE_CN = {"image": "图片", "mold": "图片发霉", "batch": "批量", "video": "视频", "camera": "摄像头"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire Recognition FireMoE GUI")
        self.resize(1320, 860)

        self.engine = None
        self.model_path: str = ""
        self.model_paths: List[str] = []
        self.frame_thread = None
        self.threadpool = QtCore.QThreadPool.globalInstance()
        self.threadpool.setMaxThreadCount(1)
        self.inference_busy = False
        self.model_loading = False
        self.model_load_thread = None
        self.model_loading_dialog = None
        self.batch_busy = False
        self.batch_paused = False
        self.batch_worker = None
        self.mold_busy = False
        self.records: List[Dict[str, Any]] = []
        self.batch_predictions: Dict[str, int] = {}
        self.batch_stats: Dict[str, Any] = {}
        self.batch_results: Dict[str, Dict[str, Any]] = {}
        self.batch_order: List[str] = []
        self.batch_current_index = -1
        self.batch_tmp_dir = ""
        self.batch_audit_path = ""
        self.batch_manual_labels_from_tmp: Dict[str, int] = {}
        self.batch_ground_truth: Dict[str, int] = {}
        self.batch_ground_truth_path = ""
        self.audit_window = None
        self.mold_source_rgb = None
        self.mold_source_path = ""
        self.mold_tmp_dir = ""
        self.mold_results: List[Dict[str, Any]] = []
        self.mold_stage_items: List[Dict[str, Any]] = []
        self.mold_current_stage_index = -1
        self.mold_current_stage_name = ""
        self.mold_current_params: Dict[str, Any] = {}
        self.mold_current_degraded_rgb = None
        self.source_type = "image"
        self.source_path = ""
        self.frame_index = 0
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.last_left: Dict[str, Any] = {"image": None, "mold": None, "batch": None, "video": None, "camera": None}
        self.last_right: Dict[str, Any] = {"image": None, "mold": None, "batch": None, "video": None, "camera": None}
        self.last_image_rgb = None
        self.last_image_maps: Dict[str, np.ndarray] = {}
        self.last_attention_overlays: Dict[str, np.ndarray] = {}
        self.latest_model_prediction = None
        self.manual_threshold = False
        self.latest_model_predictions: Dict[int, int] = {}
        self.manual_threshold_rows = set()
        self._updating_model_table = False
        self.gui_config: Dict[str, Any] = self.load_gui_config()
        self.owlvit_prompt_groups = prompt_groups_from_mapping(self.gui_config.get("owlvit_prompts", default_prompt_groups()))

        self.build_ui()
        self.restore_config_to_ui()
        self.update_buttons(False)

    def build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        model_group = QtWidgets.QGroupBox("单模型设置（FireMoE；默认读取模型阈值，可手动覆盖）")
        model_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        model_group.setMinimumHeight(BOTTOM_MODEL_BAR_HEIGHT)
        model_group.setMaximumHeight(BOTTOM_MODEL_BAR_HEIGHT)
        model_layout = QtWidgets.QVBoxLayout(model_group)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setSpacing(4)

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        self.selectModel = QtWidgets.QPushButton("📂 选择模型")
        self.selectModel.setMinimumHeight(38)
        self.loadTableModelsBtn = QtWidgets.QPushButton("🔁 加载模型")
        self.loadTableModelsBtn.setMinimumHeight(38)
        self.cuda_checkBox = QtWidgets.QCheckBox("使用 cuda 计算")
        self.cuda_checkBox.setChecked(True)
        self.modelLabel = QtWidgets.QLabel("未加载模型")
        self.modelLabel.setWordWrap(False)
        self.modelLabel.setMinimumHeight(30)
        self.modelLabel.setMaximumHeight(30)
        self.modelLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.modelLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.manualThresholdCheck = QtWidgets.QCheckBox("手动阈值")
        self.manualThresholdCheck.setChecked(False)
        self.thresholdSpin = QtWidgets.QDoubleSpinBox()
        self.thresholdSpin.setRange(0.0, 1.0)
        self.thresholdSpin.setDecimals(3)
        self.thresholdSpin.setSingleStep(0.01)
        self.thresholdSpin.setValue(DEFAULT_THRESHOLD)
        self.thresholdSpin.setMinimumWidth(90)
        self.thresholdSpin.setEnabled(False)
        self.thresholdSpin.setToolTip("未勾选手动阈值时，这里显示模型 checkpoint 自带的最佳阈值。")
        self.saveThresholdBtn = QtWidgets.QPushButton("保存阈值更改")
        self.saveThresholdBtn.setMinimumHeight(38)
        self.saveThresholdBtn.setToolTip("保存当前阈值设置；批量检测结果会立即按新阈值重新分到有火/无火列表，不重新推理。")
        self.promptSettingsBtn = QtWidgets.QPushButton("Prompt 设置")
        self.promptSettingsBtn.setMinimumHeight(38)
        self.promptStatusLabel = QtWidgets.QLabel("")
        self.promptStatusLabel.setWordWrap(False)
        self.promptStatusLabel.setMinimumWidth(185)
        self.promptStatusLabel.setMaximumHeight(30)
        self.promptStatusLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.latestProbLabel = QtWidgets.QLabel("概率：--")
        self.latestProbLabel.setWordWrap(False)
        self.latestProbLabel.setMinimumWidth(170)
        self.latestProbLabel.setMaximumHeight(30)
        self.exportBtn = QtWidgets.QPushButton("📤 导出结果")
        self.exitBtn = QtWidgets.QPushButton("⏹ 退出")
        for btn in [self.exportBtn, self.exitBtn]:
            btn.setMinimumHeight(38)

        top_layout.addWidget(self.selectModel)
        top_layout.addWidget(self.loadTableModelsBtn)
        top_layout.addWidget(self.cuda_checkBox)
        top_layout.addWidget(self.manualThresholdCheck)
        top_layout.addWidget(QtWidgets.QLabel("阈值"))
        top_layout.addWidget(self.thresholdSpin)
        top_layout.addWidget(self.saveThresholdBtn)
        top_layout.addWidget(self.promptSettingsBtn)
        top_layout.addWidget(self.promptStatusLabel)
        top_layout.addWidget(self.latestProbLabel)
        top_layout.addWidget(self.modelLabel, 1)
        top_layout.addWidget(self.exportBtn)
        top_layout.addWidget(self.exitBtn)
        model_layout.addLayout(top_layout)

        resize_layout = QtWidgets.QHBoxLayout()
        resize_layout.setContentsMargins(0, 0, 0, 0)
        resize_layout.setSpacing(8)
        self.localModelsFirstCheck = QtWidgets.QCheckBox("本地模型优先（存在则不联网）")
        self.localModelsFirstCheck.setChecked(True)
        self.localModelsFirstCheck.setToolTip(
            "默认开启。models/OWL、models/ConvNext 或 models/fire-lite.pth 存在时，"
            "直接本地加载并禁止远端校验；只有对应资源不存在时才下载。"
        )
        self.modelFilesBtn = QtWidgets.QPushButton("模型资源设置")
        self.modelFilesBtn.setToolTip("查看本地 models 路径，并可配置 fire-lite.pth 的下载地址。")
        self.resizeInputCheck = QtWidgets.QCheckBox("按高度缩小推理图像")
        self.resizeInputCheck.setChecked(DEFAULT_HEIGHT_RESIZE_ENABLED)
        self.inferHeightSpin = QtWidgets.QSpinBox()
        self.inferHeightSpin.setRange(64, 4096)
        self.inferHeightSpin.setSingleStep(32)
        self.inferHeightSpin.setValue(DEFAULT_INFER_TARGET_HEIGHT)
        self.inferHeightSpin.setMinimumWidth(86)
        self.resizeModeLabel = QtWidgets.QLabel("当前：保留原图尺寸")
        self.resizeModeLabel.setMinimumWidth(240)
        self.resizeModeLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        resize_layout.addWidget(self.localModelsFirstCheck)
        resize_layout.addWidget(self.modelFilesBtn)
        resize_layout.addWidget(self.resizeInputCheck)
        resize_layout.addWidget(QtWidgets.QLabel("目标高度"))
        resize_layout.addWidget(self.inferHeightSpin)
        resize_layout.addWidget(self.resizeModeLabel, 1)
        model_layout.addLayout(resize_layout)

        self.modelTable = QtWidgets.QTableWidget(0, 0)
        self.modelTable.hide()

        self.tabs = QtWidgets.QTabWidget()
        self.imageTab = self.build_source_tab("image", with_tile_table=True)
        self.moldTab = self.build_mold_tab()
        self.batchTab = self.build_batch_tab()
        self.videoTab = self.build_source_tab("video", with_tile_table=False)
        self.cameraTab = self.build_source_tab("camera", with_tile_table=False)
        self.tabs.addTab(self.imageTab, "图片识别")
        self.tabs.addTab(self.moldTab, "图片发霉")
        self.tabs.addTab(self.batchTab, "批量检测")
        self.tabs.addTab(self.videoTab, "视频识别")
        self.tabs.addTab(self.cameraTab, "摄像头识别")

        self.mainSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.mainSplitter.setChildrenCollapsible(False)
        self.mainSplitter.addWidget(self.tabs)
        self.mainSplitter.addWidget(model_group)
        self.mainSplitter.setStretchFactor(0, 4)
        self.mainSplitter.setStretchFactor(1, 0)
        self.mainSplitter.setSizes([760, BOTTOM_MODEL_BAR_HEIGHT])
        main_layout.addWidget(self.mainSplitter, 1)

        self.selectModel.clicked.connect(self.select_models)
        self.loadTableModelsBtn.clicked.connect(self.load_models_from_current_table)
        self.cuda_checkBox.stateChanged.connect(lambda _=None: self.save_gui_config())
        self.manualThresholdCheck.stateChanged.connect(lambda _=None: self.on_model_threshold_changed(0))
        self.thresholdSpin.valueChanged.connect(lambda _=None: self.on_model_threshold_changed(0))
        self.saveThresholdBtn.clicked.connect(self.save_threshold_changes)
        self.promptSettingsBtn.clicked.connect(self.open_prompt_settings)
        self.localModelsFirstCheck.stateChanged.connect(lambda _=None: self.save_gui_config())
        self.modelFilesBtn.clicked.connect(self.open_model_file_settings)
        self.resizeInputCheck.stateChanged.connect(lambda _=None: self.on_resize_settings_changed())
        self.inferHeightSpin.valueChanged.connect(lambda _=None: self.on_resize_settings_changed())
        self.exportBtn.clicked.connect(self.export_results)
        self.exitBtn.clicked.connect(self.close)
        self.update_threshold_controls()
        self.update_prompt_status_label()

    def build_mold_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        display_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        display_splitter.setChildrenCollapsible(False)
        left_label = self.make_display_label("左侧：发霉阶段图片")
        right_label = self.make_display_label("右侧：聚合 attention map")
        self.moldLeftLabel = left_label
        self.moldRightLabel = right_label
        display_splitter.addWidget(left_label)
        display_splitter.addWidget(right_label)
        display_splitter.setSizes([650, 650])
        layout.addWidget(display_splitter, 4)

        controls = QtWidgets.QHBoxLayout()
        self.moldOpenBtn = QtWidgets.QPushButton("选择图片")
        self.moldRunBtn = QtWidgets.QPushButton("开始发霉测试")
        self.moldExportBtn = QtWidgets.QPushButton("导出结果 CSV")
        self.moldExposureCheck = QtWidgets.QCheckBox("曝光模式")
        self.moldExposureCheck.setToolTip("勾选后，每个发霉阶段会在随机位置叠加曝光/泛白高光；采样位置会优先靠近红色、黄色分量较多的区域。")
        self.moldExposureStrengthSpin = QtWidgets.QDoubleSpinBox()
        self.moldExposureStrengthSpin.setRange(0.05, 2.50)
        self.moldExposureStrengthSpin.setDecimals(2)
        self.moldExposureStrengthSpin.setSingleStep(0.05)
        self.moldExposureStrengthSpin.setValue(0.85)
        self.moldExposureStrengthSpin.setSuffix(" x")
        self.moldExposureStrengthSpin.setToolTip("控制曝光叠加的强度，数值越大越容易出现泛白/过曝。")
        self.moldExposureSizeSpin = QtWidgets.QDoubleSpinBox()
        self.moldExposureSizeSpin.setRange(0.20, 3.00)
        self.moldExposureSizeSpin.setDecimals(2)
        self.moldExposureSizeSpin.setSingleStep(0.10)
        self.moldExposureSizeSpin.setValue(1.00)
        self.moldExposureSizeSpin.setSuffix(" x")
        self.moldExposureSizeSpin.setToolTip("控制单个曝光区域的范围大小，数值越大曝光斑块越大。")
        self.moldOverallExposureCheck = QtWidgets.QCheckBox("整体曝光模式")
        self.moldOverallExposureCheck.setToolTip("勾选后，对整张图片提升曝光；可选择全局、亮部更亮或暗部更亮。")
        self.moldOverallExposureCombo = QtWidgets.QComboBox()
        self.moldOverallExposureCombo.addItem("全局", "global")
        self.moldOverallExposureCombo.addItem("亮部更亮", "highlights")
        self.moldOverallExposureCombo.addItem("暗部更亮", "shadows")
        self.moldOverallExposureCombo.setToolTip("选择整体曝光的作用方式：全局均匀提升、偏向亮部提升或偏向暗部提升。")
        self.moldOverallExposureStrengthSpin = QtWidgets.QDoubleSpinBox()
        self.moldOverallExposureStrengthSpin.setRange(0.05, 2.50)
        self.moldOverallExposureStrengthSpin.setDecimals(2)
        self.moldOverallExposureStrengthSpin.setSingleStep(0.05)
        self.moldOverallExposureStrengthSpin.setValue(0.45)
        self.moldOverallExposureStrengthSpin.setSuffix(" x")
        self.moldOverallExposureStrengthSpin.setToolTip("控制整张图片曝光提升强度，数值越大越亮。")
        self.moldMaxStageSpin = QtWidgets.QSpinBox()
        self.moldMaxStageSpin.setRange(1, 12)
        self.moldMaxStageSpin.setValue(12)
        self.moldExposureCheck.toggled.connect(self.update_mold_exposure_controls_state)
        self.moldOverallExposureCheck.toggled.connect(self.update_mold_exposure_controls_state)
        self.moldOpenBtn.clicked.connect(self.open_mold_image)
        self.moldRunBtn.clicked.connect(self.run_mold_scan)
        self.moldExportBtn.clicked.connect(self.export_mold_results)
        controls.addWidget(self.moldOpenBtn)
        controls.addWidget(QtWidgets.QLabel("最大阶段："))
        controls.addWidget(self.moldMaxStageSpin)
        controls.addWidget(self.moldExposureCheck)
        controls.addWidget(QtWidgets.QLabel("曝光强度："))
        controls.addWidget(self.moldExposureStrengthSpin)
        controls.addWidget(QtWidgets.QLabel("曝光范围："))
        controls.addWidget(self.moldExposureSizeSpin)
        controls.addSpacing(8)
        controls.addWidget(self.moldOverallExposureCheck)
        controls.addWidget(QtWidgets.QLabel("整体方式："))
        controls.addWidget(self.moldOverallExposureCombo)
        controls.addWidget(QtWidgets.QLabel("整体强度："))
        controls.addWidget(self.moldOverallExposureStrengthSpin)
        controls.addWidget(self.moldRunBtn)
        controls.addWidget(self.moldExportBtn)
        controls.addStretch(1)
        for btn in [self.moldOpenBtn, self.moldRunBtn, self.moldExportBtn]:
            btn.setMinimumHeight(40)
        for spin in [self.moldMaxStageSpin, self.moldExposureStrengthSpin, self.moldExposureSizeSpin, self.moldOverallExposureStrengthSpin]:
            spin.setMinimumHeight(32)
            spin.setMinimumWidth(86)
        self.moldOverallExposureCombo.setMinimumHeight(32)
        self.moldOverallExposureCombo.setMinimumWidth(92)
        self.moldExposureCheck.setMinimumHeight(40)
        self.moldOverallExposureCheck.setMinimumHeight(40)
        self.update_mold_exposure_controls_state()
        layout.addLayout(controls)

        self.moldStatusLabel = QtWidgets.QLabel("等待发霉测试")
        self.moldStatusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.moldStatusLabel.setMinimumHeight(32)
        self.moldStatusLabel.setStyleSheet("font-size:16px; border:1px solid #D7E2F9; background:#FAFCFF; color:#1F2937;")
        layout.addWidget(self.moldStatusLabel)

        result_layout = QtWidgets.QHBoxLayout()
        result_label = QtWidgets.QLabel("等待预测")
        result_label.setAlignment(QtCore.Qt.AlignCenter)
        result_label.setMinimumHeight(54)
        result_label.setStyleSheet("font-size:26px; font-weight:700; border-radius:8px; background:#EEF2F7; color:#334155;")
        prob_label = QtWidgets.QLabel("概率：-- | 阈值：--")
        prob_label.setAlignment(QtCore.Qt.AlignCenter)
        prob_label.setMinimumHeight(54)
        prob_label.setStyleSheet("font-size:18px; border:1px solid #D7E2F9; background:#FAFCFF; color:#1F2937;")
        result_layout.addWidget(result_label, 1)
        result_layout.addWidget(prob_label, 2)
        layout.addLayout(result_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.moldTable = QtWidgets.QTableWidget(0, 6)
        self.moldTable.setHorizontalHeaderLabels(["阶段", "判定", "prob_fire", "阈值", "visual_base", "参数"])
        self.moldTable.horizontalHeader().setStretchLastSection(True)
        self.moldTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.moldTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.moldTable.itemSelectionChanged.connect(self.on_mold_table_selection_changed)
        self.moldSummary = QtWidgets.QTextEdit()
        self.moldSummary.setReadOnly(True)
        self.moldSummary.setPlaceholderText("用于测试同一张图片在逐级 JPEG/下采样/模糊/颜色退化后的模型判定变化；可勾选局部曝光模式或整体曝光模式进行增强测试。")
        splitter.addWidget(self.moldTable)
        splitter.addWidget(self.moldSummary)
        splitter.setSizes([760, 560])
        layout.addWidget(splitter, 3)

        self.pages["mold"] = {
            "left": left_label,
            "right": right_label,
            "open_btn": self.moldOpenBtn,
            "stop_btn": None,
            "result_label": result_label,
            "prob_label": prob_label,
            "summary": self.moldSummary,
            "tile_table": None,
            "map_labels": {},
            "batch_btn": None,
            "export_batch_btn": None,
            "batch_stats_label": None,
            "display_splitter": display_splitter,
            "detail_splitter": splitter,
        }
        return tab

    def build_source_tab(self, source_type: str, with_tile_table: bool) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        display_layout = QtWidgets.QHBoxLayout()
        display_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        display_splitter.setChildrenCollapsible(False)
        left_label = self.make_display_label(f"左侧：{self.SOURCE_CN[source_type]}原图 / 原始帧")
        right_label = self.make_display_label("右侧：聚合 attention map")
        display_splitter.addWidget(left_label)
        display_splitter.addWidget(right_label)
        display_splitter.setSizes([650, 650])
        display_layout.addWidget(display_splitter, 1)
        layout.addLayout(display_layout, 4)

        control_layout = QtWidgets.QHBoxLayout()
        batch_btn = None
        export_batch_btn = None
        batch_stats_label = None
        if source_type == "image":
            open_btn = QtWidgets.QPushButton("🖼️ 选择图片并识别")
            open_btn.clicked.connect(self.open_image)
            stop_btn = None
            control_layout.addWidget(open_btn)
        elif source_type == "video":
            open_btn = QtWidgets.QPushButton("🎞️ 选择视频并识别")
            stop_btn = QtWidgets.QPushButton("🛑 停止视频")
            open_btn.clicked.connect(self.open_video)
            stop_btn.clicked.connect(self.stop_stream)
            control_layout.addWidget(open_btn)
            control_layout.addWidget(stop_btn)
        else:
            camera_spin = QtWidgets.QSpinBox()
            camera_spin.setRange(0, 20)
            camera_spin.setValue(0)
            self.camera_id_spinBox = camera_spin
            open_btn = QtWidgets.QPushButton("📹 打开摄像头并识别")
            stop_btn = QtWidgets.QPushButton("🛑 停止摄像头")
            open_btn.clicked.connect(self.open_camera)
            stop_btn.clicked.connect(self.stop_stream)
            control_layout.addWidget(QtWidgets.QLabel("摄像头编号："))
            control_layout.addWidget(camera_spin)
            control_layout.addWidget(open_btn)
            control_layout.addWidget(stop_btn)

        for btn in [x for x in [open_btn, batch_btn, export_batch_btn, stop_btn] if x is not None]:
            btn.setMinimumHeight(40)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)

        result_layout = QtWidgets.QHBoxLayout()
        result_label = QtWidgets.QLabel("等待预测")
        result_label.setAlignment(QtCore.Qt.AlignCenter)
        result_label.setMinimumHeight(54)
        result_label.setStyleSheet("font-size:26px; font-weight:700; border-radius:8px; background:#EEF2F7; color:#334155;")
        prob_label = QtWidgets.QLabel("概率：-- | 阈值：--")
        prob_label.setAlignment(QtCore.Qt.AlignCenter)
        prob_label.setMinimumHeight(54)
        prob_label.setStyleSheet("font-size:18px; border:1px solid #D7E2F9; background:#FAFCFF; color:#1F2937;")
        result_layout.addWidget(result_label, 1)
        result_layout.addWidget(prob_label, 2)
        layout.addLayout(result_layout)

        if with_tile_table:
            detail_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            detail_splitter.setChildrenCollapsible(False)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            summary = QtWidgets.QTextEdit()
            summary.setReadOnly(True)
            summary.setMinimumHeight(210)
            summary.setPlaceholderText("图片调试信息：单模型概率、阈值、global/local attention、专家路由权重、OWL-ViT 语义先验等。")
            tile_table = QtWidgets.QTableWidget(0, 3)
            tile_table.setHorizontalHeaderLabels(["专家", "路由占比", "专家概率"])
            tile_table.horizontalHeader().setStretchLastSection(True)
            tile_table.setMinimumHeight(210)
            splitter.addWidget(summary)
            splitter.addWidget(tile_table)
            splitter.setSizes([560, 760])
            detail_splitter.addWidget(splitter)

            maps_scroll = QtWidgets.QScrollArea()
            maps_scroll.setWidgetResizable(True)
            maps_scroll.setMinimumHeight(230)
            maps_container = QtWidgets.QWidget()
            maps_layout = QtWidgets.QGridLayout(maps_container)
            maps_layout.setContentsMargins(6, 6, 6, 6)
            maps_layout.setSpacing(8)
            map_labels: Dict[str, QtWidgets.QLabel] = {}
            map_titles = [
                ("global", "Global attention map"),
                ("local", "Local attention map"),
                ("semantic", "Semantic attention map"),
                ("mix", "Mix attention map"),
            ]
            for idx, (key, title) in enumerate(map_titles):
                box = QtWidgets.QFrame()
                box.setFrameShape(QtWidgets.QFrame.StyledPanel)
                box_layout = QtWidgets.QVBoxLayout(box)
                title_label = QtWidgets.QLabel(title)
                title_label.setAlignment(QtCore.Qt.AlignCenter)
                title_label.setStyleSheet("font-weight:700; color:#1F2937;")
                img_label = ClickableImageLabel("等待图片预测", image_key=key)
                img_label.setAlignment(QtCore.Qt.AlignCenter)
                img_label.setMinimumSize(240, 150)
                img_label.setToolTip("点击放大；放大窗口中可保存 PNG 或用系统图片查看器打开")
                img_label.setStyleSheet("border:1px solid #D7E2F9; background:#FAFCFF; color:#64748B;")
                img_label.clicked.connect(self.open_attention_preview)
                box_layout.addWidget(title_label)
                box_layout.addWidget(img_label)
                row, col = divmod(idx, 2)
                maps_layout.addWidget(box, row, col)
                map_labels[key] = img_label
            maps_scroll.setWidget(maps_container)
            detail_splitter.addWidget(maps_scroll)
            detail_splitter.setSizes([230, 260])
            layout.addWidget(detail_splitter, 3)
        else:
            summary = QtWidgets.QTextEdit()
            summary.setReadOnly(True)
            summary.setMinimumHeight(160)
            summary.setPlaceholderText(f"{self.SOURCE_CN[source_type]}状态区：显示单模型预测结果、概率和专家路由摘要。")
            tile_table = None
            layout.addWidget(summary, 1)

        self.pages[source_type] = {
            "left": left_label,
            "right": right_label,
            "open_btn": open_btn,
            "stop_btn": stop_btn,
            "result_label": result_label,
            "prob_label": prob_label,
            "summary": summary,
            "tile_table": tile_table,
            "map_labels": locals().get("map_labels", {}),
            "batch_btn": batch_btn,
            "export_batch_btn": export_batch_btn,
            "batch_stats_label": batch_stats_label,
            "display_splitter": display_splitter,
            "detail_splitter": locals().get("detail_splitter", None),
        }
        return tab

    def build_batch_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        display_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        display_splitter.setChildrenCollapsible(False)
        left_label = ClickableImageLabel("左侧：批量图片预览", image_key="batch_current")
        left_label.setAlignment(QtCore.Qt.AlignCenter)
        left_label.setMinimumSize(520, 320)
        left_label.setToolTip("点击用系统图片查看器打开；勾选 attention 时会并排打开 attention 叠加图")
        left_label.setStyleSheet("border:1px solid #D7E2F9; background:#FAFCFF; color:#4A5568;")
        left_label.clicked.connect(self.open_batch_current_external)
        right_label = ClickableImageLabel("右侧：批量聚合 attention map", image_key="mix")
        right_label.setAlignment(QtCore.Qt.AlignCenter)
        right_label.setMinimumSize(520, 320)
        right_label.setToolTip("点击放大 attention；也可在弹窗中保存 PNG 或用系统图片查看器打开")
        right_label.setStyleSheet("border:1px solid #D7E2F9; background:#FAFCFF; color:#4A5568;")
        right_label.clicked.connect(self.open_attention_preview)
        display_splitter.addWidget(left_label)
        display_splitter.addWidget(right_label)
        display_splitter.setSizes([650, 650])
        layout.addWidget(display_splitter, 4)

        control_layout = QtWidgets.QHBoxLayout()
        select_folder_btn = QtWidgets.QPushButton("选择文件夹并自动检测")
        pause_batch_btn = QtWidgets.QPushButton("暂停批量")
        resume_batch_btn = QtWidgets.QPushButton("继续批量")
        stop_batch_btn = QtWidgets.QPushButton("停止批量")
        include_attention_check = QtWidgets.QCheckBox("缓存 attention map 到硬盘（默认关闭，仍会显示当前图）")
        include_attention_check.setChecked(False)
        batch_size_spin = QtWidgets.QSpinBox()
        batch_size_spin.setRange(1, 9999)
        batch_size_spin.setValue(16)
        batch_size_spin.setMinimumWidth(82)
        batch_size_spin.setToolTip("每批送入检测的图片数量；已有 tmp 结果会优先读取。")
        prev_btn = QtWidgets.QPushButton("上一张")
        next_btn = QtWidgets.QPushButton("下一张")
        open_current_btn = QtWidgets.QPushButton("打开当前图片")
        mark_fire_btn = QtWidgets.QPushButton("人工标为有火")
        mark_no_fire_btn = QtWidgets.QPushButton("人工标为无火")
        restore_btn = QtWidgets.QPushButton("恢复模型判断")
        audit_btn = QtWidgets.QPushButton("审计窗口")
        import_gt_btn = QtWidgets.QPushButton("导入对比JSON")
        export_gt_template_btn = QtWidgets.QPushButton("导出对比模板")
        export_json_btn = QtWidgets.QPushButton("导出 JSON")
        export_split_images_btn = QtWidgets.QPushButton("导出有火/无火图片")
        export_meta_btn = QtWidgets.QPushButton("导出 Meta JSON")

        select_folder_btn.clicked.connect(self.open_image_folder_batch)
        pause_batch_btn.clicked.connect(self.pause_batch_detection)
        resume_batch_btn.clicked.connect(self.resume_batch_detection)
        stop_batch_btn.clicked.connect(self.stop_batch_detection)
        prev_btn.clicked.connect(self.show_batch_prev)
        next_btn.clicked.connect(self.show_batch_next)
        open_current_btn.clicked.connect(self.open_batch_current_external)
        mark_fire_btn.clicked.connect(lambda _=None: self.set_batch_manual_label(1))
        mark_no_fire_btn.clicked.connect(lambda _=None: self.set_batch_manual_label(0))
        restore_btn.clicked.connect(lambda _=None: self.set_batch_manual_label(None))
        audit_btn.clicked.connect(self.open_batch_audit_window)
        import_gt_btn.clicked.connect(self.import_batch_ground_truth_json)
        export_gt_template_btn.clicked.connect(self.export_batch_ground_truth_template)
        export_json_btn.clicked.connect(self.export_batch_json)
        export_split_images_btn.clicked.connect(self.export_batch_split_images)
        export_meta_btn.clicked.connect(self.export_batch_meta_json)
        for btn in [
            select_folder_btn, pause_batch_btn, resume_batch_btn, stop_batch_btn, prev_btn, next_btn, open_current_btn,
            mark_fire_btn, mark_no_fire_btn, restore_btn, audit_btn, import_gt_btn,
            export_gt_template_btn, export_json_btn, export_split_images_btn, export_meta_btn,
        ]:
            btn.setMinimumHeight(36)

        control_layout.addWidget(select_folder_btn)
        control_layout.addWidget(pause_batch_btn)
        control_layout.addWidget(resume_batch_btn)
        control_layout.addWidget(stop_batch_btn)
        control_layout.addWidget(include_attention_check)
        control_layout.addWidget(QtWidgets.QLabel("每批"))
        control_layout.addWidget(batch_size_spin)
        control_layout.addStretch(1)
        control_layout.addWidget(prev_btn)
        control_layout.addWidget(next_btn)
        control_layout.addWidget(open_current_btn)
        control_layout.addWidget(mark_fire_btn)
        control_layout.addWidget(mark_no_fire_btn)
        control_layout.addWidget(restore_btn)
        control_layout.addWidget(audit_btn)
        control_layout.addWidget(import_gt_btn)
        control_layout.addWidget(export_gt_template_btn)
        control_layout.addWidget(export_json_btn)
        control_layout.addWidget(export_split_images_btn)
        control_layout.addWidget(export_meta_btn)
        layout.addLayout(control_layout)

        progress_layout = QtWidgets.QHBoxLayout()
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        folder_label = QtWidgets.QLabel("未选择文件夹")
        folder_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        progress_layout.addWidget(progress_bar, 2)
        progress_layout.addWidget(folder_label, 3)
        layout.addLayout(progress_layout)

        result_layout = QtWidgets.QHBoxLayout()
        result_label = QtWidgets.QLabel("等待批量检测")
        result_label.setAlignment(QtCore.Qt.AlignCenter)
        result_label.setMinimumHeight(50)
        result_label.setStyleSheet("font-size:24px; font-weight:700; border-radius:8px; background:#EEF2F7; color:#334155;")
        prob_label = QtWidgets.QLabel("统计：--")
        prob_label.setAlignment(QtCore.Qt.AlignCenter)
        prob_label.setMinimumHeight(50)
        prob_label.setStyleSheet("font-size:17px; border:1px solid #D7E2F9; background:#FAFCFF; color:#1F2937;")
        accuracy_label = QtWidgets.QLabel("准确率：--")
        accuracy_label.setAlignment(QtCore.Qt.AlignCenter)
        accuracy_label.setMinimumHeight(50)
        accuracy_label.setMinimumWidth(210)
        accuracy_label.setStyleSheet("font-size:17px; font-weight:700; border:1px solid #D7E2F9; background:#F8FAFC; color:#1F2937;")
        result_layout.addWidget(result_label, 1)
        result_layout.addWidget(prob_label, 2)
        result_layout.addWidget(accuracy_label, 1)
        layout.addLayout(result_layout)

        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom_splitter.setChildrenCollapsible(False)
        fire_list = QtWidgets.QListWidget()
        no_fire_list = QtWidgets.QListWidget()
        for lst in [fire_list, no_fire_list]:
            lst.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            lst.itemClicked.connect(self.on_batch_item_clicked)
            lst.itemDoubleClicked.connect(lambda _item: self.open_batch_current_external())

        lists_widget = QtWidgets.QWidget()
        lists_layout = QtWidgets.QHBoxLayout(lists_widget)
        fire_group = QtWidgets.QGroupBox("有火 / 疑似")
        fire_layout = QtWidgets.QVBoxLayout(fire_group)
        fire_layout.addWidget(fire_list)
        no_fire_group = QtWidgets.QGroupBox("无火")
        no_fire_layout = QtWidgets.QVBoxLayout(no_fire_group)
        no_fire_layout.addWidget(no_fire_list)
        lists_layout.addWidget(fire_group)
        lists_layout.addWidget(no_fire_group)

        summary = QtWidgets.QTextEdit()
        summary.setReadOnly(True)
        summary.setMinimumHeight(180)
        summary.setPlaceholderText("批量状态区：显示当前文件、单模型概率、人工审计修改、meta 路径和统计结果。")
        bottom_splitter.addWidget(lists_widget)
        bottom_splitter.addWidget(summary)
        bottom_splitter.setSizes([620, 680])
        layout.addWidget(bottom_splitter, 3)

        self.pages["batch"] = {
            "left": left_label,
            "right": right_label,
            "open_btn": select_folder_btn,
            "pause_btn": pause_batch_btn,
            "resume_btn": resume_batch_btn,
            "stop_btn": stop_batch_btn,
            "result_label": result_label,
            "prob_label": prob_label,
            "summary": summary,
            "tile_table": None,
            "map_labels": {},
            "batch_btn": select_folder_btn,
            "export_batch_btn": export_json_btn,
            "export_split_images_btn": export_split_images_btn,
            "export_meta_btn": export_meta_btn,
            "batch_stats_label": prob_label,
            "accuracy_label": accuracy_label,
            "display_splitter": display_splitter,
            "detail_splitter": bottom_splitter,
            "progress_bar": progress_bar,
            "folder_label": folder_label,
            "include_attention_check": include_attention_check,
            "batch_size_spin": batch_size_spin,
            "fire_list": fire_list,
            "no_fire_list": no_fire_list,
            "prev_btn": prev_btn,
            "next_btn": next_btn,
            "open_current_btn": open_current_btn,
            "mark_fire_btn": mark_fire_btn,
            "mark_no_fire_btn": mark_no_fire_btn,
            "restore_btn": restore_btn,
            "audit_btn": audit_btn,
            "import_gt_btn": import_gt_btn,
            "export_gt_template_btn": export_gt_template_btn,
            "export_split_images_btn": export_split_images_btn,
        }
        return tab

    def make_display_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(560, 360)
        label.setStyleSheet("border:1px solid #D7E2F9; background:#FAFCFF; color:#4A5568;")
        return label

    def load_gui_config(self) -> Dict[str, Any]:
        if not CONFIG_PATH.exists():
            return {}
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.exception("读取 GUI 配置失败 | path=%s", CONFIG_PATH)
            return {}

    def restore_config_to_ui(self):
        cfg = self.gui_config or {}
        self.cuda_checkBox.setChecked(bool(cfg.get("use_cuda", True)))
        asset_cfg = cfg.get("model_files") if isinstance(cfg.get("model_files"), dict) else {}
        self.localModelsFirstCheck.setChecked(bool(asset_cfg.get("local_first", True)))
        resize_cfg = cfg.get("inference_resize") if isinstance(cfg.get("inference_resize"), dict) else {}
        self.resizeInputCheck.setChecked(bool(resize_cfg.get(
            "height_resize_enabled",
            resize_cfg.get("resize_enabled", resize_cfg.get("enabled", DEFAULT_HEIGHT_RESIZE_ENABLED)),
        )))
        self.inferHeightSpin.setValue(int(resize_cfg.get("target_height", DEFAULT_INFER_TARGET_HEIGHT)))
        self.update_resize_controls()
        if hasattr(self, "camera_id_spinBox"):
            self.camera_id_spinBox.setValue(int(cfg.get("camera_id", 0)))
            self.camera_id_spinBox.valueChanged.connect(lambda _=None: self.save_gui_config())
        saved_model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        if not saved_model:
            saved_models = [m for m in cfg.get("models", []) if isinstance(m, dict) and m.get("path")]
            saved_model = saved_models[0] if saved_models else {}
        saved_path = str(saved_model.get("path", ""))
        local_first = bool(asset_cfg.get("local_first", True))
        fire_lite_url = str(asset_cfg.get("fire_lite_url", "")).strip()
        if local_first and DEFAULT_FIRE_LITE_PATH.is_file():
            path = str(DEFAULT_FIRE_LITE_PATH)
        elif local_first and fire_lite_url:
            # Keep the canonical target selected; the actual download starts only when loading.
            path = str(DEFAULT_FIRE_LITE_PATH)
        elif saved_path and Path(saved_path).is_file():
            path = saved_path
        elif DEFAULT_FIRE_LITE_PATH.is_file():
            path = str(DEFAULT_FIRE_LITE_PATH)
        else:
            path = ""
        if path:
            self.model_path = path
            self.model_paths = [path]
            if Path(path).is_file():
                self.modelLabel.setText(f"已恢复模型：{Path(path).name}（未加载权重，点击‘加载模型’）")
            else:
                self.modelLabel.setText(f"主权重缺失：加载时将下载到 {DEFAULT_FIRE_LITE_PATH}")
            manual_threshold = bool(saved_model.get("manual_threshold", False))
            self.manualThresholdCheck.setChecked(manual_threshold)
            self.thresholdSpin.setValue(float(saved_model.get("threshold", DEFAULT_THRESHOLD)) if manual_threshold else float(DEFAULT_THRESHOLD))
            self.update_threshold_controls()
        sizes = cfg.get("main_splitter_sizes")
        if isinstance(sizes, list) and len(sizes) == 2:
            self.mainSplitter.setSizes([int(sizes[0]), int(sizes[1])])

    def save_gui_config(self):
        try:
            cfg = {
                "use_cuda": bool(self.cuda_checkBox.isChecked()) if hasattr(self, "cuda_checkBox") else True,
                "camera_id": int(self.camera_id_spinBox.value()) if hasattr(self, "camera_id_spinBox") else 0,
                "model": self.get_model_settings()[0] if self.get_model_settings() else {},
                "model_files": self.get_model_file_settings(),
                "inference_resize": self.get_resize_settings() if hasattr(self, "resizeInputCheck") else {},
                "owlvit_prompts": {key: list(values) for key, values in self.owlvit_prompt_groups.items()},
                "main_splitter_sizes": [int(x) for x in self.mainSplitter.sizes()] if hasattr(self, "mainSplitter") else [],
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            self.gui_config = cfg
        except Exception:
            logger.exception("保存 GUI 配置失败 | path=%s", CONFIG_PATH)

    def update_prompt_status_label(self):
        if not hasattr(self, "promptStatusLabel"):
            return
        groups = prompt_groups_from_mapping(self.owlvit_prompt_groups)
        self.owlvit_prompt_groups = groups
        total = sum(len(values) for values in groups.values())
        self.promptStatusLabel.setText(f"Prompt：3 组 / {total} 条")
        self.promptStatusLabel.setToolTip(
            "当前 OWL-ViT prompt 数量："
            f"正向 {len(groups['positive'])}，负向 {len(groups['negative'])}，疑似 {len(groups['suspicious'])}。"
        )

    def open_prompt_settings(self):
        dialog = PromptSettingsDialog(self.owlvit_prompt_groups, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.owlvit_prompt_groups = dialog.prompt_groups()
        self.update_prompt_status_label()
        self.save_gui_config()
        self.force_reinfer_current()

    def force_reinfer_current(self):
        if self.engine is None:
            return
        if self.inference_busy or self.batch_busy or self.mold_busy:
            QtWidgets.QMessageBox.information(self, "正在识别", "当前任务还在运行，prompt 已保存；本轮结束后请再次点击确认或重新开始检测。")
            return
        current_widget = self.tabs.currentWidget() if hasattr(self, "tabs") else None
        if current_widget is self.imageTab:
            source_type = "image"
        elif current_widget is self.videoTab:
            source_type = "video"
        elif current_widget is self.cameraTab:
            source_type = "camera"
        elif current_widget is self.moldTab:
            self.moldSummary.append("\nPrompt 已保存。请点击“开始发霉测试”重新按阶段识别。")
            return
        elif current_widget is self.batchTab:
            self.force_reinfer_current_batch_item()
            return
        else:
            source_type = self.source_type if self.source_type in ["image", "video", "camera"] else "image"
        rgb = self.last_left.get(source_type)
        if rgb is None:
            page = self.pages.get(source_type, {})
            summary = page.get("summary")
            if summary is not None:
                summary.setPlainText("Prompt 已保存。当前页面还没有可重新识别的图片或帧。")
            return
        meta = {
            "source_type": source_type,
            "source_path": self.source_path,
            "frame_index": self.frame_index if source_type in ["video", "camera"] else 0,
        }
        self.clear_attention_display(source_type)
        self.submit_inference(np.ascontiguousarray(rgb.copy()), meta)

    def force_reinfer_current_batch_item(self):
        key = self.current_batch_key()
        if key is None or key not in self.batch_results:
            summary = self.pages.get("batch", {}).get("summary")
            if summary is not None:
                summary.append("\nPrompt 已保存。批量页当前没有选中样本；重新选择文件夹时会用新 prompt 重新生成 attention map。")
            return
        item = self.batch_results[key]
        path = str(item.get("path", ""))
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "当前图片不存在", f"无法重新识别当前批量图片：\n{path}")
            return
        rgb = imread_rgb(path)
        self.clear_attention_display("batch")
        self.submit_inference(rgb, {
            "source_type": "batch",
            "source_path": path,
            "batch_key": key,
            "batch_root": self.batch_stats.get("root_dir", ""),
            "frame_index": self.batch_current_index + 1,
        })

    def get_model_file_settings(self) -> Dict[str, Any]:
        previous = (self.gui_config or {}).get("model_files", {})
        if not isinstance(previous, dict):
            previous = {}
        return {
            "local_first": bool(self.localModelsFirstCheck.isChecked())
            if hasattr(self, "localModelsFirstCheck") else True,
            "models_dir": str(MODELS_DIR),
            "fire_lite_url": str(previous.get("fire_lite_url", "")).strip(),
        }

    def open_model_file_settings(self):
        current = self.get_model_file_settings()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("模型资源设置")
        dialog.resize(680, 210)
        layout = QtWidgets.QVBoxLayout(dialog)

        rule_label = QtWidgets.QLabel(
            "本地优先规则：只要对应文件夹/文件存在，就直接离线加载，不进行任何网上校验。"
        )
        rule_label.setWordWrap(True)
        layout.addWidget(rule_label)

        form = QtWidgets.QFormLayout()
        models_label = QtWidgets.QLineEdit(str(MODELS_DIR))
        models_label.setReadOnly(True)
        url_edit = QtWidgets.QLineEdit(str(current.get("fire_lite_url", "")))
        url_edit.setPlaceholderText("可选：fire-lite.pth 的 http(s) 下载地址")
        form.addRow("models 文件夹", models_label)
        form.addRow("fire-lite.pth 下载地址", url_edit)
        layout.addLayout(form)

        note = QtWidgets.QLabel(
            "OWL 缺失时从 checkpoint 中的 Hugging Face 模型名下载到 models/OWL；"
            "ConvNext 缺失时下载到 models/ConvNext。fire-lite.pth 的来源由此地址指定。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.gui_config = dict(self.gui_config or {})
        asset_cfg = dict(self.gui_config.get("model_files", {}) or {})
        asset_cfg["local_first"] = bool(self.localModelsFirstCheck.isChecked())
        asset_cfg["models_dir"] = str(MODELS_DIR)
        asset_cfg["fire_lite_url"] = url_edit.text().strip()
        self.gui_config["model_files"] = asset_cfg
        self.save_gui_config()

    def saved_setting_for_path(self, path: str) -> Dict[str, Any]:
        target = str(Path(path))
        model = (self.gui_config or {}).get("model", {})
        if isinstance(model, dict) and str(model.get("path", "")) == target:
            return dict(model)
        for item in (self.gui_config or {}).get("models", []):
            if str(item.get("path", "")) == target:
                return dict(item)
        return {}


    def update_mold_exposure_controls_state(self, checked=None):
        local_enabled = bool(getattr(self, "moldExposureCheck", None) and self.moldExposureCheck.isChecked())
        overall_enabled = bool(getattr(self, "moldOverallExposureCheck", None) and self.moldOverallExposureCheck.isChecked())
        busy = bool(getattr(self, "batch_busy", False) or getattr(self, "mold_busy", False))
        for widget in [
            getattr(self, "moldExposureStrengthSpin", None),
            getattr(self, "moldExposureSizeSpin", None),
        ]:
            if widget is not None:
                widget.setEnabled(local_enabled and not busy)
        for widget in [
            getattr(self, "moldOverallExposureCombo", None),
            getattr(self, "moldOverallExposureStrengthSpin", None),
        ]:
            if widget is not None:
                widget.setEnabled(overall_enabled and not busy)

    def update_buttons(self, model_loaded: bool):
        busy = bool(self.batch_busy or self.mold_busy or self.model_loading)
        for source_type in ["image", "video", "camera"]:
            page = self.pages.get(source_type, {})
            if page.get("open_btn") is not None:
                page["open_btn"].setEnabled(model_loaded and not busy)
            if page.get("stop_btn") is not None:
                page["stop_btn"].setEnabled(False)
            if page.get("batch_btn") is not None:
                page["batch_btn"].setEnabled(model_loaded and not busy)
            if page.get("export_batch_btn") is not None:
                page["export_batch_btn"].setEnabled(bool(self.batch_predictions) and not busy)
        batch_page = self.pages.get("batch", {})
        if batch_page:
            has_batch = bool(self.batch_results)
            if batch_page.get("open_btn") is not None:
                batch_page["open_btn"].setEnabled(model_loaded and not busy)
            if batch_page.get("pause_btn") is not None:
                batch_page["pause_btn"].setEnabled(self.batch_busy and not bool(getattr(self, "batch_paused", False)))
            if batch_page.get("resume_btn") is not None:
                batch_page["resume_btn"].setEnabled(self.batch_busy and bool(getattr(self, "batch_paused", False)))
            if batch_page.get("stop_btn") is not None:
                batch_page["stop_btn"].setEnabled(self.batch_busy)
            for key in ["prev_btn", "next_btn", "open_current_btn", "mark_fire_btn", "mark_no_fire_btn", "restore_btn", "audit_btn"]:
                if batch_page.get(key) is not None:
                    batch_page[key].setEnabled(has_batch)
            if batch_page.get("export_gt_template_btn") is not None:
                batch_page["export_gt_template_btn"].setEnabled(has_batch and not busy)
            if batch_page.get("export_batch_btn") is not None:
                batch_page["export_batch_btn"].setEnabled(bool(self.batch_predictions) and not busy)
            if batch_page.get("export_split_images_btn") is not None:
                batch_page["export_split_images_btn"].setEnabled(bool(self.batch_results) and not busy)
            if batch_page.get("export_meta_btn") is not None:
                batch_page["export_meta_btn"].setEnabled(bool(self.batch_results) and not busy)
        if hasattr(self, "moldOpenBtn"):
            self.moldOpenBtn.setEnabled(model_loaded and not busy)
            self.moldRunBtn.setEnabled(model_loaded and not busy)
            self.moldExportBtn.setEnabled(bool(self.mold_results) and not busy)
            if hasattr(self, "moldExposureCheck"):
                self.moldExposureCheck.setEnabled(not busy)
                if hasattr(self, "moldOverallExposureCheck"):
                    self.moldOverallExposureCheck.setEnabled(not busy)
                self.update_mold_exposure_controls_state()
        if hasattr(self, "saveThresholdBtn"):
            self.saveThresholdBtn.setEnabled(bool(self.model_path))
        self.loadTableModelsBtn.setEnabled(bool(self.model_paths) and not self.model_loading)
        self.selectModel.setEnabled(not self.model_loading)
        self.cuda_checkBox.setEnabled(not self.model_loading)
        self.localModelsFirstCheck.setEnabled(not self.model_loading)
        self.exportBtn.setEnabled(True)

    def make_check_item(self, checked: bool) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item

    def make_text_item(self, text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(str(text))
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        return item

    def selected_model_rows(self) -> List[int]:
        rows = sorted({idx.row() for idx in self.modelTable.selectionModel().selectedRows()})
        if not rows:
            row = self.modelTable.currentRow()
            if row >= 0:
                rows = [row]
        return rows

    def show_model_table_context_menu(self, pos):
        row = self.modelTable.rowAt(pos.y())
        if row >= 0 and not self.modelTable.selectionModel().isRowSelected(row, QtCore.QModelIndex()):
            self.modelTable.selectRow(row)

        rows = self.selected_model_rows()
        if not rows:
            return

        menu = QtWidgets.QMenu(self)
        remove_action = menu.addAction(f"从列表移除选中模型（{len(rows)}个）")
        delete_action = menu.addAction("从磁盘删除选中模型文件...")
        action = menu.exec_(self.modelTable.viewport().mapToGlobal(pos))
        if action == remove_action:
            self.remove_model_rows(rows, delete_files=False)
        elif action == delete_action:
            self.remove_model_rows(rows, delete_files=True)

    def remove_model_rows(self, rows: List[int], delete_files: bool = False):
        rows_set = {int(r) for r in rows if 0 <= int(r) < len(self.model_paths)}
        rows = sorted(rows_set, reverse=True)
        if not rows:
            return

        current_settings = self.get_model_settings()
        remaining_settings = [dict(s) for i, s in enumerate(current_settings) if i not in rows_set]
        paths = [self.model_paths[r] for r in rows]
        if delete_files:
            msg = "将从模型列表移除，并尝试删除磁盘上的模型文件：\n\n" + "\n".join(paths[:12])
            if len(paths) > 12:
                msg += f"\n... 还有 {len(paths) - 12} 个"
            msg += "\n\n这个操作不可从 GUI 撤销，确定继续？"
            if QtWidgets.QMessageBox.question(self, "确认删除模型文件", msg) != QtWidgets.QMessageBox.Yes:
                return

        failed = []
        for row in rows:
            path = self.model_paths[row]
            if delete_files:
                try:
                    Path(path).unlink(missing_ok=True)
                except TypeError:
                    try:
                        p = Path(path)
                        if p.exists():
                            p.unlink()
                    except Exception as e:
                        failed.append(f"{path}: {e}")
                except Exception as e:
                    failed.append(f"{path}: {e}")

            del self.model_paths[row]
            if self.engine is not None:
                for attr in ["model_paths", "models", "model_names", "model_thresholds", "model_threshold_sources"]:
                    values = getattr(self.engine, attr, None)
                    if values is not None and row < len(values):
                        del values[row]

        old_rows_left = [i for i in range(len(self.model_paths) + len(rows)) if i not in rows_set]
        self.manual_threshold_rows = {new_row for new_row, old_row in enumerate(old_rows_left) if old_row in self.manual_threshold_rows}
        self.latest_model_predictions = {
            new_row: self.latest_model_predictions[old_row]
            for new_row, old_row in enumerate(old_rows_left)
            if old_row in self.latest_model_predictions
        }

        self.gui_config = dict(self.gui_config or {})
        self.gui_config["models"] = remaining_settings
        self.populate_model_table(self.model_paths)
        if self.engine is None or not self.model_paths:
            self.engine = None
            self.modelLabel.setText("未加载模型" if not self.model_paths else f"模型列表剩余 {len(self.model_paths)} 个（未加载）")
            self.update_buttons(False)
        else:
            self.modelLabel.setText(f"已加载 {len(self.model_paths)} 个模型")
            self.apply_model_thresholds_from_engine()
            self.update_buttons(True)
        self.save_gui_config()

        if failed:
            QtWidgets.QMessageBox.warning(self, "部分模型文件删除失败", "\n".join(failed[:12]))

    def select_models(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择模型权重", "", "PyTorch Weights (*.pth *.pt)")
        if not path:
            return
        self.stop_stream()
        self.engine = None
        self.model_path = path
        self.model_paths = [path]
        self.latest_model_prediction = None
        saved = self.saved_setting_for_path(path)
        manual_threshold = bool(saved.get("manual_threshold", False))
        self.manualThresholdCheck.setChecked(manual_threshold)
        self.thresholdSpin.setValue(float(saved.get("threshold", DEFAULT_THRESHOLD)) if manual_threshold else float(DEFAULT_THRESHOLD))
        self.update_threshold_controls()
        self.modelLabel.setText(f"已选择模型：{Path(path).name}（未加载）")
        self.save_gui_config()
        self.load_models_from_current_table()

    def load_models_from_current_table(self):
        if self.model_loading:
            if self.model_loading_dialog is not None:
                self.model_loading_dialog.show()
                self.model_loading_dialog.raise_()
                self.model_loading_dialog.activateWindow()
            return
        if self.inference_busy or self.batch_busy or self.mold_busy:
            QtWidgets.QMessageBox.information(self, "任务正在运行", "请先等待当前识别任务完成，再加载模型。")
            return

        asset_cfg = self.get_model_file_settings()
        local_first = bool(asset_cfg.get("local_first", True))
        use_cuda = self.cuda_checkBox.isChecked() and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        selected_path = self.model_path or None

        self.stop_stream()
        self.model_loading = True
        self.loadTableModelsBtn.setText("⏳ 正在加载...")
        self.modelLabel.setText(f"正在加载模型 | device={device}")
        self.pages["image"]["summary"].setPlainText(
            "正在解析模型资源并加载权重，界面仍可正常响应。\n"
            + (selected_path or str(DEFAULT_FIRE_LITE_PATH))
        )
        self.update_buttons(self.engine is not None)

        dialog = QtWidgets.QProgressDialog(
            "正在加载模型，请稍候…\n模型解析、必要资源下载和权重初始化正在后台执行。",
            "",
            0,
            0,
            self,
        )
        dialog.setWindowTitle("正在加载")
        dialog.setCancelButton(None)
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        dialog.setMinimumWidth(460)
        self.model_loading_dialog = dialog
        dialog.show()

        thread = ModelLoadThread(
            selected_path=selected_path,
            device=device,
            local_models_first=local_first,
            models_dir=MODELS_DIR,
            download_url=str(asset_cfg.get("fire_lite_url", "")),
            parent=self,
        )
        self.model_load_thread = thread
        thread.loaded.connect(self.on_model_load_success)
        thread.error.connect(self.on_model_load_error)
        thread.finished.connect(lambda t=thread: self.on_model_load_thread_finished(t))
        thread.start()

    def _finish_model_loading_ui(self):
        self.model_loading = False
        self.loadTableModelsBtn.setText("🔁 加载模型")
        if self.model_loading_dialog is not None:
            self.model_loading_dialog.close()
            self.model_loading_dialog.deleteLater()
            self.model_loading_dialog = None

    def on_model_load_success(self, engine, resolved_path: str, device_name: str):
        self.engine = engine
        self.model_path = str(resolved_path)
        self.model_paths = [self.model_path]
        self._finish_model_loading_ui()
        self.apply_model_thresholds_from_engine()
        self.modelLabel.setText(
            f"已加载：{Path(self.model_path).name} | device={device_name} | 配置={CONFIG_PATH.name}"
        )
        for source_type in ["image", "mold", "batch", "video", "camera"]:
            self.pages[source_type]["summary"].setPlainText("模型加载完成。请选择输入开始识别。")
            self.set_result_banner(source_type, None)
        self.update_buttons(True)
        self.save_gui_config()

    def on_model_load_error(self, msg: str):
        self._finish_model_loading_ui()
        logger.error("模型加载失败：%s", msg)
        self.modelLabel.setText("模型加载失败")
        self.pages["image"]["summary"].setPlainText("模型加载失败：\n" + msg)
        self.update_buttons(self.engine is not None)
        QtWidgets.QMessageBox.critical(
            self,
            "模型加载失败",
            msg + "\n\n完整异常堆栈已输出到控制台。",
        )

    def on_model_load_thread_finished(self, thread):
        if self.model_load_thread is thread:
            self.model_load_thread = None
        thread.deleteLater()

    def populate_model_table(self, paths: List[str]):
        self._updating_model_table = True
        self.modelTable.blockSignals(True)
        try:
            self.manual_threshold_rows = set()
            self.modelTable.setRowCount(len(paths))
            for row, path in enumerate(paths):
                name = Path(path).name
                saved = self.saved_setting_for_path(path)
                enabled = bool(saved.get("enabled", True))
                manual_threshold = bool(saved.get("manual_threshold", False))
                if manual_threshold:
                    self.manual_threshold_rows.add(row)
                threshold = float(saved.get("threshold", DEFAULT_THRESHOLD)) if manual_threshold else float(DEFAULT_THRESHOLD)
                self.modelTable.setItem(row, 0, self.make_check_item(enabled))
                self.modelTable.setItem(row, 1, self.make_text_item(str(row)))
                self.modelTable.setItem(row, 2, self.make_text_item(name))
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0.0, 1.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.01)
                spin.setValue(threshold)
                spin.setAlignment(QtCore.Qt.AlignCenter)
                spin.valueChanged.connect(lambda _=None, r=row: self.on_model_threshold_changed(r))
                self.modelTable.setCellWidget(row, 3, spin)
                self.modelTable.setItem(row, 4, self.make_text_item("--"))
                source_label = "手动阈值" if manual_threshold else "待加载模型阈值"
                self.modelTable.setItem(row, 5, self.make_text_item(source_label if enabled else "未启用"))
                self.modelTable.setItem(row, 6, self.make_text_item(str(path)))
                self.set_model_row_color(row, "disabled" if not enabled else None)
            self.modelTable.resizeColumnsToContents()
            self.modelTable.setColumnHidden(6, True)
        finally:
            self.modelTable.blockSignals(False)
            self._updating_model_table = False
        self.refresh_model_table_styles()

    def apply_model_thresholds_from_engine(self):
        if self.engine is None:
            return
        model_threshold = float(getattr(self.engine, "model_threshold", DEFAULT_THRESHOLD))
        if not self.manualThresholdCheck.isChecked():
            self.thresholdSpin.blockSignals(True)
            self.thresholdSpin.setValue(model_threshold)
            self.thresholdSpin.blockSignals(False)
        self.update_threshold_controls()
        self.latestProbLabel.setText(f"概率：-- | 阈值源：{getattr(self.engine, 'model_threshold_source', 'model')}")

    def get_model_settings(self) -> List[Dict[str, Any]]:
        if not self.model_path:
            return []
        model_threshold = float(getattr(self.engine, "model_threshold", DEFAULT_THRESHOLD)) if self.engine is not None else float(DEFAULT_THRESHOLD)
        threshold_source = str(getattr(self.engine, "model_threshold_source", "model")) if self.engine is not None else "config"
        manual_threshold = bool(self.manualThresholdCheck.isChecked())
        threshold = float(self.thresholdSpin.value()) if manual_threshold else float(model_threshold)
        prompt_groups = prompt_groups_from_mapping(self.owlvit_prompt_groups)
        return [{
            "model_index": 0,
            "name": Path(self.model_path).name,
            "path": str(self.model_path),
            "enabled": True,
            "threshold": float(threshold),
            "model_threshold": float(model_threshold),
            "manual_threshold": bool(manual_threshold),
            "threshold_source": str("manual" if manual_threshold else threshold_source),
            "owl_positive_prompts": list(prompt_groups["positive"]),
            "owl_negative_prompts": list(prompt_groups["negative"]),
            "owl_suspicious_prompts": list(prompt_groups["suspicious"]),
        }]

    def get_resize_settings(self) -> Dict[str, Any]:
        return {
            "height_resize_enabled": bool(self.resizeInputCheck.isChecked()) if hasattr(self, "resizeInputCheck") else DEFAULT_HEIGHT_RESIZE_ENABLED,
            "target_height": int(self.inferHeightSpin.value()) if hasattr(self, "inferHeightSpin") else DEFAULT_INFER_TARGET_HEIGHT,
        }

    def update_resize_controls(self):
        enabled = bool(self.resizeInputCheck.isChecked()) if hasattr(self, "resizeInputCheck") else False
        for widget in [
            getattr(self, "inferHeightSpin", None),
        ]:
            if widget is not None:
                widget.setEnabled(enabled)
        if hasattr(self, "resizeModeLabel"):
            if not enabled:
                text = "当前：保留原图尺寸"
            else:
                text = f"当前：若原图高度大于 {self.inferHeightSpin.value()}，按高度等比缩小；小图保持原尺寸"
            self.resizeModeLabel.setText(text)

    def on_resize_settings_changed(self):
        self.update_resize_controls()
        self.save_gui_config()

    def update_threshold_controls(self):
        manual = bool(self.manualThresholdCheck.isChecked()) if hasattr(self, "manualThresholdCheck") else False
        if hasattr(self, "thresholdSpin"):
            self.thresholdSpin.setEnabled(manual)
            if manual:
                self.thresholdSpin.setToolTip("手动阈值已启用：输入框中的值会覆盖模型自带最佳阈值。")
            else:
                self.thresholdSpin.setToolTip("正在使用模型 checkpoint 自带的最佳阈值；勾选手动阈值后可编辑。")
        if hasattr(self, "manualThresholdCheck"):
            self.manualThresholdCheck.setToolTip("取消勾选时自动使用模型自带最佳阈值；勾选后使用输入框中的阈值。")

    def on_model_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._updating_model_table:
            return
        if item is not None and item.column() == 0:
            self.refresh_model_table_styles()
            self.save_gui_config()

    def on_model_threshold_changed(self, row: int):
        if self._updating_model_table:
            return
        self.manual_threshold = bool(self.manualThresholdCheck.isChecked())
        if not self.manual_threshold and self.engine is not None:
            self.thresholdSpin.blockSignals(True)
            self.thresholdSpin.setValue(float(getattr(self.engine, "model_threshold", DEFAULT_THRESHOLD)))
            self.thresholdSpin.blockSignals(False)
        self.update_threshold_controls()
        self.save_gui_config()

    def save_threshold_changes(self):
        self.manual_threshold = bool(self.manualThresholdCheck.isChecked())
        if not self.manual_threshold and self.engine is not None:
            self.thresholdSpin.blockSignals(True)
            self.thresholdSpin.setValue(float(getattr(self.engine, "model_threshold", DEFAULT_THRESHOLD)))
            self.thresholdSpin.blockSignals(False)
        self.update_threshold_controls()
        self.save_gui_config()
        if self.batch_worker is not None:
            try:
                self.batch_worker.model_settings = [dict(x) for x in self.get_model_settings()]
            except Exception:
                pass
        changed = self.apply_threshold_to_batch_results()
        if changed > 0:
            self.pages["batch"]["summary"].append(
                f"\n阈值更改已保存，已按当前阈值重新更新批量有火/无火列表：{changed} 张。"
            )
        elif self.batch_results:
            self.pages["batch"]["summary"].append("\n阈值更改已保存，批量列表无需变化。")
        else:
            self.latestProbLabel.setText(f"概率：-- | 阈值：{float(self.thresholdSpin.value()):.3f}")

    def current_model_threshold_context(self) -> Tuple[List[Dict[str, Any]], float, str]:
        settings = self.get_model_settings()
        model_threshold = float(getattr(self.engine, "model_threshold", DEFAULT_THRESHOLD)) if self.engine is not None else float(DEFAULT_THRESHOLD)
        threshold_source = str(getattr(self.engine, "model_threshold_source", "model")) if self.engine is not None else "config"
        return settings, model_threshold, threshold_source

    def apply_threshold_to_batch_results(self) -> int:
        if not self.batch_results:
            return 0
        settings, model_threshold, threshold_source = self.current_model_threshold_context()
        changed = 0
        current_key = self.current_batch_key()
        for key, item in self.batch_results.items():
            old_final = self.batch_final_label(item)
            old_pred = int(item.get("pred_label", 0))
            refreshed = reclassify_result_by_threshold(
                item.get("result", {}), settings, model_threshold, threshold_source
            )
            new_pred = int(BatchInferenceRunnable._binary_export_label(refreshed))
            item["result"] = refreshed
            item["pred_label"] = new_pred
            if old_pred != new_pred or old_final != self.batch_final_label(item):
                changed += 1
        self.refresh_batch_lists()
        self.batch_predictions = self.make_batch_predictions_from_current_labels()
        self.batch_stats["predictions"] = dict(self.batch_predictions)
        self.update_batch_stats_label()
        if current_key in self.batch_results:
            self.select_batch_key(current_key)
        self.save_batch_audit_tmp()
        if self.audit_window is not None and self.audit_window.isVisible():
            self.audit_window.refresh_from_parent()
        self.update_buttons(self.engine is not None)
        return changed

    def refresh_batch_lists(self):
        page = self.pages.get("batch", {})
        for list_key in ["fire_list", "no_fire_list"]:
            lst = page.get(list_key)
            if lst is not None:
                lst.blockSignals(True)
                lst.clear()
                lst.blockSignals(False)
        for key in self.batch_order:
            if key in self.batch_results:
                self.add_or_update_batch_list_item(key)

    def refresh_model_table_styles(self):
        for row in range(self.modelTable.rowCount()):
            enabled_item = self.modelTable.item(row, 0)
            enabled = enabled_item is None or enabled_item.checkState() == QtCore.Qt.Checked
            if not enabled:
                pred = "disabled"
            else:
                pred = self.latest_model_predictions.get(row, None)
            self.set_model_row_color(row, pred)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "", IMAGE_EXTS)
        if not path:
            return
        self.stop_stream()
        rgb = imread_rgb(path)
        self.source_type = "image"
        self.source_path = path
        self.frame_index = 0
        self.tabs.setCurrentWidget(self.imageTab)
        self.show_left(rgb, "image")
        self.submit_inference(rgb, {"source_type": "image", "source_path": path, "frame_index": 0})

    @staticmethod
    def mold_degrade_rgb(rgb: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        return mold_degrade_rgb(rgb, params)

    def open_mold_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择发霉测试图片", "", IMAGE_EXTS)
        if not path:
            return
        self.stop_stream()
        self.mold_source_rgb = imread_rgb(path)
        self.mold_source_path = path
        self.mold_tmp_dir = ""
        self.mold_results = []
        self.mold_stage_items = []
        self.mold_current_stage_index = -1
        self.moldTable.setRowCount(0)
        self.moldSummary.setPlainText(f"已选择图片：{path}\n点击“开始发霉测试”后会按 stage_00-stage_12 逐级退化并推理；可勾选“曝光模式”叠加局部高光，或勾选“整体曝光模式”提升整张图曝光。")
        self.moldStatusLabel.setText("已选择图片，等待测试")
        self.set_result_banner("mold", None)
        self.show_left(self.mold_source_rgb, "mold")
        self.last_right["mold"] = None
        self.moldRightLabel.clear()
        self.moldRightLabel.setText("等待发霉测试")
        self.tabs.setCurrentWidget(self.moldTab)

    def prepare_mold_tmp_dir(self) -> Path:
        source_name = safe_artifact_stem(Path(self.mold_source_path).stem or "image")
        digest = hashlib.sha1(str(self.mold_source_path).encode("utf-8")).hexdigest()[:10]
        tmp_dir = BATCH_TMP_ROOT / f"mold_cache_{time.strftime('%Y%m%d_%H%M%S')}_{source_name}_{digest}"
        (tmp_dir / "stage_images").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "overlays").mkdir(parents=True, exist_ok=True)
        self.mold_tmp_dir = str(tmp_dir)
        return tmp_dir

    @staticmethod
    def load_mold_artifact(path: str) -> np.ndarray | None:
        if not path or not Path(path).exists():
            return None
        try:
            return imread_rgb(path)
        except Exception:
            return None

    def run_mold_scan(self):
        if self.engine is None:
            QtWidgets.QMessageBox.warning(self, "未加载模型", "请先加载模型权重。")
            return
        if self.mold_busy:
            QtWidgets.QMessageBox.information(self, "发霉测试中", "当前发霉测试仍在运行，请等待完成。")
            return
        if self.batch_busy:
            QtWidgets.QMessageBox.information(self, "批量检测中", "当前批量检测仍在运行，请稍后再开始发霉测试。")
            return
        if self.inference_busy:
            QtWidgets.QMessageBox.information(self, "正在识别", "当前单张/视频识别任务还在运行，请稍后再开始发霉测试。")
            return
        if self.mold_source_rgb is None:
            self.open_mold_image()
            if self.mold_source_rgb is None:
                return
        self.mold_results = []
        self.mold_current_stage_index = -1
        self.mold_current_stage_name = ""
        self.mold_current_params = {}
        self.mold_current_degraded_rgb = None
        self.moldTable.setRowCount(0)
        self.last_left["mold"] = None
        self.last_right["mold"] = None
        self.moldLeftLabel.clear()
        self.moldLeftLabel.setText("准备生成发霉阶段图片")
        self.moldRightLabel.clear()
        self.moldRightLabel.setText("发霉测试运行中，完成后选择阶段查看 attention")
        self.moldRunBtn.setEnabled(False)
        self.moldExportBtn.setEnabled(False)
        max_stage = int(self.moldMaxStageSpin.value())
        settings = self.get_model_settings()
        if not any(s.get("enabled", True) for s in settings):
            QtWidgets.QMessageBox.warning(self, "未加载模型", "请先加载模型。")
            self.moldRunBtn.setEnabled(True)
            return
        total = min(max_stage, len(MOLD_STAGE_PRESETS) - 1) + 1
        self.prepare_mold_tmp_dir()
        self.mold_stage_items = [dict(item) for item in MOLD_STAGE_PRESETS[:total]]
        exposure_enabled = bool(getattr(self, "moldExposureCheck", None) and self.moldExposureCheck.isChecked())
        overall_exposure_enabled = bool(getattr(self, "moldOverallExposureCheck", None) and self.moldOverallExposureCheck.isChecked())
        exposure_strength = float(self.moldExposureStrengthSpin.value()) if hasattr(self, "moldExposureStrengthSpin") else 0.85
        exposure_size_scale = float(self.moldExposureSizeSpin.value()) if hasattr(self, "moldExposureSizeSpin") else 1.0
        overall_exposure_strength = float(self.moldOverallExposureStrengthSpin.value()) if hasattr(self, "moldOverallExposureStrengthSpin") else 0.45
        overall_exposure_strategy = "global"
        overall_exposure_label = "全局"
        if hasattr(self, "moldOverallExposureCombo"):
            overall_exposure_strategy = str(self.moldOverallExposureCombo.currentData() or "global")
            overall_exposure_label = str(self.moldOverallExposureCombo.currentText() or "全局")
        exposure_run_seed = int(time.time_ns() & 0xFFFFFFFF)
        if exposure_enabled or overall_exposure_enabled:
            for idx, item in enumerate(self.mold_stage_items):
                params = dict(item.get("params", {}))
                if overall_exposure_enabled:
                    params.update({
                        "overall_exposure_mode": True,
                        "overall_exposure_strategy": overall_exposure_strategy,
                        "overall_exposure_label": overall_exposure_label,
                        "overall_exposure_strength": round(overall_exposure_strength, 3),
                    })
                if exposure_enabled:
                    params.update({
                        "exposure_mode": True,
                        "exposure_seed": int((exposure_run_seed + idx * 7919) & 0xFFFFFFFF),
                        "exposure_strength": round(exposure_strength, 3),
                        "exposure_size_scale": round(exposure_size_scale, 3),
                        "exposure_spots": int(min(4, max(1, 1 + idx // 4))),
                        "exposure_warm_bias": 0.78,
                    })
                item["params"] = params
        self.mold_busy = True
        self.source_type = "mold"
        self.tabs.setCurrentWidget(self.moldTab)
        mode_parts = []
        if exposure_enabled:
            mode_parts.append(f"局部曝光 强度{exposure_strength:.2f}x 范围{exposure_size_scale:.2f}x")
        if overall_exposure_enabled:
            mode_parts.append(f"整体曝光 {overall_exposure_label} 强度{overall_exposure_strength:.2f}x")
        mode_text = " | " + "；".join(mode_parts) if mode_parts else ""
        self.moldStatusLabel.setText(f"开始发霉测试：0/{total}{mode_text}")
        self.update_buttons(self.engine is not None)
        self.start_next_mold_stage()

    def start_next_mold_stage(self):
        if not self.mold_busy:
            return
        next_index = self.mold_current_stage_index + 1
        total = len(self.mold_stage_items)
        if next_index >= total:
            self.finish_mold_scan()
            return

        preset = self.mold_stage_items[next_index]
        name = str(preset.get("name", f"stage_{next_index:02d}"))
        params = dict(preset.get("params", {}))
        try:
            degraded_rgb = self.mold_degrade_rgb(self.mold_source_rgb, params)
        except Exception as exc:
            logger.exception("发霉测试图像处理失败 | stage=%s", name)
            self.finish_mold_scan(error_msg=f"{name}: {exc}")
            return

        self.mold_current_stage_index = next_index
        self.mold_current_stage_name = name
        self.mold_current_params = params
        self.mold_current_degraded_rgb = np.ascontiguousarray(degraded_rgb.copy())
        self.moldStatusLabel.setText(f"正在逐张测试：{next_index + 1}/{total} | {name}")
        self.show_left(degraded_rgb, "mold")
        self.clear_attention_display("mold")
        self.submit_inference(degraded_rgb, {
            "source_type": "mold",
            "source_path": f"{self.mold_source_path}::{name}",
            "frame_index": next_index,
            "mold_stage": name,
            "mold_params": params,
            "_mold_internal": True,
        })
        if not self.inference_busy:
            self.finish_mold_scan(error_msg=f"{name}: 无法启动推理任务")

    def handle_mold_inference_result(
        self,
        result: Dict[str, Any],
        maps: Dict[str, np.ndarray],
        infer_rgb: np.ndarray,
        overlay: np.ndarray,
    ):
        name = str(result.get("mold_stage", self.mold_current_stage_name))
        params = dict(result.get("mold_params", self.mold_current_params))
        stage_rgb = self.mold_current_degraded_rgb
        if stage_rgb is None:
            stage_rgb = infer_rgb
        row = {
            "stage": name,
            "params": params,
            "rgb_path": "",
            "overlay_path": "",
            "rgb": np.ascontiguousarray(stage_rgb.copy()),
            "overlay": np.ascontiguousarray(overlay.copy()),
            "result": dict(result),
        }
        self.mold_results.append(row)
        self.append_mold_table_row(row)
        table_row = self.moldTable.rowCount() - 1
        if table_row >= 0:
            self.moldTable.selectRow(table_row)
            self.show_mold_result(table_row)

        total = len(self.mold_stage_items)
        self.moldStatusLabel.setText(
            f"已完成：{self.mold_current_stage_index + 1}/{total} | {name} | {result.get('result_cn', '')}"
        )
        QtCore.QTimer.singleShot(0, self.start_next_mold_stage)

    def finish_mold_scan(self, error_msg: str | None = None):
        self.mold_busy = False
        self.update_buttons(self.engine is not None)
        total = len(self.mold_stage_items)
        done = len(self.mold_results)
        fire_count = sum(
            1 for item in self.mold_results
            if str(item.get("result", {}).get("result", "")) == "fire"
        )
        if error_msg:
            self.moldStatusLabel.setText(f"发霉测试中断：已完成 {done}/{total}，有火 {fire_count}")
            self.moldSummary.append("\n发霉测试错误：" + str(error_msg))
        else:
            self.moldStatusLabel.setText(f"发霉测试完成：{fire_count}/{total} 个阶段判为有火")
        if self.mold_results and self.moldTable.currentRow() < 0:
            self.moldTable.selectRow(0)
            self.show_mold_result(0)

    def append_mold_table_row(self, row: Dict[str, Any]):
        result = row.get("result", {})
        table = self.moldTable
        r = table.rowCount()
        table.insertRow(r)
        values = [
            row.get("stage", ""),
            result.get("result_cn", result.get("result", "")),
            f"{float(result.get('prob_fire', 0.0)):.4f}",
            f"{float(result.get('threshold', DEFAULT_THRESHOLD)):.3f}",
            f"{float(result.get('visual_base_logit', 0.0)):.3f}",
            json.dumps(row.get("params", {}), ensure_ascii=False),
        ]
        for c, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value))
            if str(result.get("result", "")) == "fire":
                item.setBackground(QtGui.QColor("#FEE2E2"))
            else:
                item.setBackground(QtGui.QColor("#DCFCE7"))
            table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def on_mold_table_selection_changed(self):
        rows = self.moldTable.selectionModel().selectedRows() if self.moldTable.selectionModel() else []
        if not rows:
            return
        self.show_mold_result(int(rows[0].row()))

    def show_mold_result(self, index: int):
        if index < 0 or index >= len(self.mold_results):
            return
        item = self.mold_results[index]
        result = item.get("result", {})
        rgb = item.get("rgb")
        overlay = item.get("overlay")
        if rgb is None:
            rgb = self.load_mold_artifact(str(item.get("rgb_path", "")))
        if overlay is None:
            overlay = self.load_mold_artifact(str(item.get("overlay_path", "")))
        if rgb is not None:
            self.last_left["mold"] = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
            self.moldLeftLabel.setPixmap(rgb_to_qpixmap(rgb, self.moldLeftLabel, smooth=True))
        else:
            self.moldLeftLabel.setText("当前阶段图片缓存不存在")
        if overlay is not None:
            self.last_right["mold"] = np.ascontiguousarray(np.asarray(overlay, dtype=np.uint8).copy())
            self.moldRightLabel.setPixmap(rgb_to_qpixmap(overlay, self.moldRightLabel, smooth=True))
        else:
            self.moldRightLabel.setText("当前阶段 overlay 缓存不存在")
        lines = [
            f"阶段：{item.get('stage', '')}",
            f"参数：{json.dumps(item.get('params', {}), ensure_ascii=False)}",
            f"判定：{result.get('result_cn', '')} | prob_fire={float(result.get('prob_fire', 0.0)):.4f} | threshold={float(result.get('threshold', DEFAULT_THRESHOLD)):.3f}",
            f"visual_base_logit={float(result.get('visual_base_logit', 0.0)):.3f}, raw_logit={float(result.get('raw_logit', 0.0)):.3f}, firearbiter_delta={float(result.get('firearbiter_delta_logit', 0.0)):.3f}",
            f"fire_evidence={float(result.get('fire_evidence_logit', 0.0)):.3f}, local_peak={float(result.get('local_evidence_peak_logit', 0.0)):.3f}, negative_weight={float(result.get('negative_weight', 0.0)):.3f}",
            f"owl_margin={float(result.get('owl_margin', 0.0)):.3f}, region_owl_margin={float(result.get('region_owl_margin', 0.0)):.3f}",
            f"缓存目录：{self.mold_tmp_dir}",
        ]
        self.moldSummary.setPlainText("\n".join(lines))

    def export_mold_results(self):
        if not self.mold_results:
            QtWidgets.QMessageBox.information(self, "没有结果", "请先完成一次发霉测试。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出发霉测试 CSV", "mold_test_results.csv", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "stage", "result", "prob_fire", "threshold", "visual_base_logit", "raw_logit",
                "firearbiter_delta_logit", "fire_evidence_logit", "local_evidence_peak_logit",
                "negative_weight", "owl_margin", "region_owl_margin", "params_json",
            ])
            writer.writeheader()
            for item in self.mold_results:
                result = item.get("result", {})
                writer.writerow({
                    "stage": item.get("stage", ""),
                    "result": result.get("result", ""),
                    "prob_fire": result.get("prob_fire", ""),
                    "threshold": result.get("threshold", ""),
                    "visual_base_logit": result.get("visual_base_logit", ""),
                    "raw_logit": result.get("raw_logit", ""),
                    "firearbiter_delta_logit": result.get("firearbiter_delta_logit", ""),
                    "fire_evidence_logit": result.get("fire_evidence_logit", ""),
                    "local_evidence_peak_logit": result.get("local_evidence_peak_logit", ""),
                    "negative_weight": result.get("negative_weight", ""),
                    "owl_margin": result.get("owl_margin", ""),
                    "region_owl_margin": result.get("region_owl_margin", ""),
                    "params_json": json.dumps(item.get("params", {}), ensure_ascii=False),
                })

    @staticmethod
    def collect_image_paths(folder: str) -> List[str]:
        root = Path(folder)
        paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        return sorted(paths)

    def open_image_folder_batch(self):
        if self.engine is None:
            QtWidgets.QMessageBox.information(self, "未加载模型", "请先加载模型。")
            return
        if self.batch_busy:
            QtWidgets.QMessageBox.information(self, "批量检测中", "当前批量检测仍在运行，可以先停止后再重新选择文件夹。")
            return
        if self.mold_busy:
            QtWidgets.QMessageBox.information(self, "发霉测试中", "当前发霉测试仍在运行，请稍后再开始批量检测。")
            return
        if self.inference_busy:
            QtWidgets.QMessageBox.information(self, "正在识别", "当前单张/视频识别任务还在运行，请稍后再开始批量检测。")
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择需要批量检测的图片文件夹")
        if not folder:
            return
        image_paths = self.collect_image_paths(folder)
        if not image_paths:
            QtWidgets.QMessageBox.information(self, "没有图片", "该文件夹下没有找到 jpg/png/bmp/webp/tif 图片。")
            return

        model_settings = self.get_model_settings()
        if not any(s.get("enabled", True) for s in model_settings):
            QtWidgets.QMessageBox.warning(self, "未加载模型", "请先加载模型。")
            return

        self.stop_stream()
        self.source_type = "batch"
        self.tabs.setCurrentWidget(self.batchTab)
        resize_settings = self.get_resize_settings()
        self.batch_tmp_dir = str(batch_cache_dir(folder, model_settings, resize_settings))
        Path(self.batch_tmp_dir).mkdir(parents=True, exist_ok=True)
        self.batch_audit_path = str(Path(self.batch_tmp_dir) / BATCH_AUDIT_FILENAME)
        self.batch_busy = True
        self.batch_paused = False
        self.batch_worker = None
        self.batch_predictions = {}
        self.batch_results = {}
        self.batch_order = []
        self.batch_current_index = -1
        self.load_batch_audit_tmp()
        batch_size = int(self.pages["batch"]["batch_size_spin"].value())
        self.batch_stats = {
            "total": len(image_paths),
            "done": 0,
            "fire": 0,
            "no_fire": 0,
            "suspect": 0,
            "failed": 0,
            "cached": 0,
            "batch_size": batch_size,
            "predictions": {},
            "errors": [],
            "root_dir": folder,
            "tmp_dir": self.batch_tmp_dir,
            "stopped": False,
        }
        self.pages["batch"]["fire_list"].clear()
        self.pages["batch"]["no_fire_list"].clear()
        self.pages["batch"]["progress_bar"].setRange(0, len(image_paths))
        self.pages["batch"]["progress_bar"].setValue(0)
        self.pages["batch"]["folder_label"].setText(f"文件夹：{folder} | tmp：{self.batch_tmp_dir}")
        self.update_batch_stats_label()
        self.update_buttons(self.engine is not None)
        self.pages["batch"]["summary"].setPlainText(
            f"开始批量检测：{folder}\n图片数量：{len(image_paths)}\n每批送入：{batch_size}\n"
            f"tmp 缓存：{self.batch_tmp_dir}\n正在处理第 1 张..."
        )

        include_attention = bool(self.pages["batch"]["include_attention_check"].isChecked())
        worker = BatchInferenceRunnable(
            self.engine,
            image_paths=image_paths,
            model_settings=model_settings,
            root_dir=folder,
            save_attention=include_attention,
            resize_settings=resize_settings,
            tmp_dir=self.batch_tmp_dir,
            batch_size=batch_size,
        )
        self.batch_worker = worker
        worker.signals.progress.connect(self.on_batch_progress)
        worker.signals.error.connect(self.on_batch_error)
        worker.signals.finished.connect(self.on_batch_finished)
        self.threadpool.start(worker)

    def update_batch_stats_label(self):
        label = self.pages.get("batch", {}).get("batch_stats_label")
        accuracy_label = self.pages.get("batch", {}).get("accuracy_label")
        stats = self.recompute_batch_stats()
        text = (
            f"统计：{stats['done']}/{stats['total']} | 有火={stats['fire']} | 无火={stats['no_fire']} | "
            f"疑似={stats['suspect']} | 失败={stats['failed']} | 人工修改={stats['manual_changed']}"
        )
        if self.batch_stats:
            text += f" | 缓存={int(self.batch_stats.get('cached', 0))}"
        acc = self.compute_batch_accuracy()
        if acc.get("compared", 0) > 0:
            text += f" | 对比准确率={acc['accuracy']:.2%} ({acc['correct']}/{acc['compared']})"
        if label is not None:
            label.setText(text)
        if accuracy_label is not None:
            if acc.get("compared", 0) > 0:
                accuracy_label.setText(f"准确率：{acc['accuracy']:.2%} ({acc['correct']}/{acc['compared']})")
                accuracy_label.setToolTip(f"对比 JSON：{acc.get('ground_truth_path', '')}；未匹配：{acc.get('missing', 0)}")
            elif self.batch_ground_truth:
                accuracy_label.setText("准确率：未匹配")
                accuracy_label.setToolTip(f"已导入对比 JSON，但当前结果未匹配到文件名：{self.batch_ground_truth_path}")
            else:
                accuracy_label.setText("准确率：--")
                accuracy_label.setToolTip("导入对比 JSON 后实时显示准确率。")

    def recompute_batch_stats(self) -> Dict[str, int]:
        base = self.batch_stats or {}
        total = int(base.get("total", len(self.batch_order)))
        failed = int(base.get("failed", 0))
        done = len(self.batch_results)
        fire = 0
        no_fire = 0
        suspect = 0
        manual_changed = 0
        audited = 0
        model_match_manual = 0
        for item in self.batch_results.values():
            result = item.get("result", {})
            if result.get("result") == "suspect":
                suspect += 1
            final_label = self.batch_final_label(item)
            fire += 1 if final_label == 1 else 0
            no_fire += 1 if final_label == 0 else 0
            if item.get("manual_label") is not None:
                audited += 1
                if int(item.get("manual_label")) != int(item.get("pred_label", 0)):
                    manual_changed += 1
                else:
                    model_match_manual += 1
        return {
            "total": total,
            "done": done,
            "fire": fire,
            "no_fire": no_fire,
            "suspect": suspect,
            "failed": failed,
            "manual_changed": manual_changed,
            "audited": audited,
            "model_match_manual": model_match_manual,
        }

    def compute_batch_accuracy(self) -> Dict[str, Any]:
        labels = self.batch_ground_truth or {}
        if not labels:
            return {"compared": 0, "correct": 0, "accuracy": None}
        compared = 0
        correct = 0
        missing = 0
        for key in self.batch_order:
            item = self.batch_results.get(key)
            if not item:
                continue
            truth = ground_truth_label_for_key(labels, key, item.get("path", ""))
            if truth is None:
                missing += 1
                continue
            compared += 1
            if int(truth) == int(self.batch_final_label(item)):
                correct += 1
        accuracy = (float(correct) / float(compared)) if compared else None
        return {
            "compared": compared,
            "correct": correct,
            "missing": missing,
            "accuracy": accuracy,
            "ground_truth_path": self.batch_ground_truth_path,
        }

    @staticmethod
    def batch_final_label(item: Dict[str, Any]) -> int:
        if item.get("manual_label") is not None:
            return int(item.get("manual_label"))
        return int(item.get("pred_label", 0))

    def make_batch_predictions_from_current_labels(self) -> Dict[str, int]:
        return {key: self.batch_final_label(self.batch_results[key]) for key in self.batch_order if key in self.batch_results}

    def load_batch_audit_tmp(self):
        self.batch_manual_labels_from_tmp = {}
        if not self.batch_audit_path or not Path(self.batch_audit_path).exists():
            return
        try:
            with open(self.batch_audit_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            labels = payload.get("manual_labels", payload if isinstance(payload, dict) else {})
            if isinstance(labels, dict):
                for key, value in labels.items():
                    label = normalize_label_value(value)
                    if label is not None:
                        self.batch_manual_labels_from_tmp[str(key).replace("\\", "/")] = int(label)
        except Exception as e:
            self.pages["batch"]["summary"].append(f"\n读取历史审计 JSON 失败：{e}")

    def save_batch_audit_tmp(self):
        if not self.batch_tmp_dir:
            return
        tmp_dir = Path(self.batch_tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.batch_audit_path = str(tmp_dir / BATCH_AUDIT_FILENAME)
        manual_labels = {
            key: int(item.get("manual_label"))
            for key, item in self.batch_results.items()
            if item.get("manual_label") is not None
        }
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_dir": self.batch_stats.get("root_dir", ""),
            "tmp_dir": self.batch_tmp_dir,
            "manual_labels": manual_labels,
            "predictions": self.make_batch_predictions_from_current_labels(),
        }
        with open(self.batch_audit_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        pred_path = tmp_dir / BATCH_CURRENT_PREDICTIONS_FILENAME
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(payload["predictions"], f, ensure_ascii=False, separators=(",", ":"))

    def batch_mix_attention_rgb(self, item: Dict[str, Any]) -> np.ndarray | None:
        artifacts = item.get("artifacts", {})
        overlay_path = artifacts.get("overlay_paths", {}).get("mix") if isinstance(artifacts, dict) else None
        if overlay_path and Path(overlay_path).exists():
            try:
                return imread_rgb(overlay_path)
            except Exception:
                pass
        rgb = item.get("rgb")
        maps = item.get("maps", {})
        if rgb is None:
            try:
                rgb = prepare_rgb_for_inference(imread_rgb(item["path"]), **self.get_resize_settings())[0]
            except Exception:
                return None
        if isinstance(maps, dict) and "mix" in maps:
            return heatmap_overlay(np.asarray(rgb, dtype=np.uint8), maps["mix"])
        attention_path = artifacts.get("attention_paths", {}).get("mix") if isinstance(artifacts, dict) else None
        if attention_path and Path(attention_path).exists():
            try:
                gray = cv2.imdecode(np.fromfile(str(attention_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    return heatmap_overlay(np.asarray(rgb, dtype=np.uint8), gray.astype(np.float32) / 255.0)
            except Exception:
                pass
        return None

    def open_batch_audit_window(self):
        if not self.batch_results:
            QtWidgets.QMessageBox.information(self, "没有批量结果", "请先完成一次文件夹批量检测。")
            return
        if self.audit_window is None:
            self.audit_window = BatchAuditWindow(self)
        self.audit_window.refresh_from_parent()
        self.audit_window.show()
        self.audit_window.raise_()
        self.audit_window.activateWindow()

    def import_batch_ground_truth_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "导入对比 JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            labels = parse_ground_truth_payload(payload)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导入失败", str(e))
            return
        if not labels:
            QtWidgets.QMessageBox.warning(self, "没有可用标签", "对比 JSON 中没有识别到 filename/path/key 与有火标签。")
            return
        self.batch_ground_truth = labels
        self.batch_ground_truth_path = path
        self.update_batch_stats_label()
        if self.audit_window is not None and self.audit_window.isVisible():
            self.audit_window.refresh_from_parent()
        acc = self.compute_batch_accuracy()
        if acc.get("compared", 0) > 0:
            msg = f"已导入 {len(labels)} 条标签。\n当前可对比 {acc['compared']} 张，准确率 {acc['accuracy']:.2%}。"
        else:
            msg = f"已导入 {len(labels)} 条标签，但当前批量结果还没有匹配到文件名。"
        QtWidgets.QMessageBox.information(self, "导入完成", msg)

    def export_batch_ground_truth_template(self):
        if not self.batch_order:
            QtWidgets.QMessageBox.information(self, "没有批量结果", "请先完成一次文件夹批量检测。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出对比 JSON 模板", "fire_ground_truth_template.json", "JSON Files (*.json)")
        if not path:
            return
        payload = {
            "description": "将每个文件名对应的 null 改为 1/0、true/false、或 有火/无火 后，可导入用于计算准确率。",
            "items": {key: None for key in self.batch_order},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(self, "导出完成", f"已导出：\n{path}")

    def export_batch_audit_json(self):
        if not self.batch_results:
            QtWidgets.QMessageBox.information(self, "没有批量结果", "请先完成一次文件夹批量检测。")
            return
        self.save_batch_audit_tmp()
        default_name = "fire_batch_audit_labels.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存审计 JSON", default_name, "JSON Files (*.json)")
        if not path:
            return
        with open(self.batch_audit_path, "r", encoding="utf-8") as src:
            payload = json.load(src)
        with open(path, "w", encoding="utf-8") as dst:
            json.dump(payload, dst, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(self, "保存完成", f"已保存：\n{path}")

    def add_or_update_batch_list_item(self, key: str):
        item = self.batch_results.get(key)
        if item is None:
            return
        page = self.pages["batch"]
        for list_key in ["fire_list", "no_fire_list"]:
            lst = page[list_key]
            for row in range(lst.count() - 1, -1, -1):
                row_item = lst.item(row)
                if row_item.data(QtCore.Qt.UserRole) == key:
                    lst.takeItem(row)

        target = page["fire_list"] if self.batch_final_label(item) == 1 else page["no_fire_list"]
        row_item = QtWidgets.QListWidgetItem(key)
        row_item.setData(QtCore.Qt.UserRole, key)
        if item.get("manual_label") is not None:
            row_item.setText(f"{key}  [人工]")
        target.addItem(row_item)

    def select_batch_key(self, key: str):
        if key not in self.batch_results:
            return
        try:
            self.batch_current_index = self.batch_order.index(key)
        except ValueError:
            self.batch_current_index = -1
        for list_key in ["fire_list", "no_fire_list"]:
            lst = self.pages["batch"][list_key]
            lst.blockSignals(True)
            lst.clearSelection()
            for i in range(lst.count()):
                row_item = lst.item(i)
                if row_item.data(QtCore.Qt.UserRole) == key:
                    row_item.setSelected(True)
                    lst.setCurrentItem(row_item)
                    break
            lst.blockSignals(False)
        self.display_batch_item(key)
        if self.audit_window is not None and self.audit_window.isVisible():
            self.audit_window.refresh_from_parent()

    def on_batch_progress(self, idx: int, total: int, path: str, result: Dict[str, Any], maps: Dict[str, np.ndarray], rgb: np.ndarray):
        key = str(result.get("batch_key") or BatchInferenceRunnable._json_key(path, self.batch_stats.get("root_dir", "")))
        pred_label = BatchInferenceRunnable._binary_export_label(result)
        self.batch_predictions[key] = int(pred_label)
        self.batch_results[key] = {
            "key": key,
            "path": str(path),
            "pred_label": int(pred_label),
            "manual_label": self.batch_manual_labels_from_tmp.get(key),
            "result": dict(result),
            "maps": {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()},
            "rgb": np.ascontiguousarray(rgb.copy()),
            "artifacts": dict(result.get("batch_artifacts", {})),
        }
        if key not in self.batch_order:
            self.batch_order.append(key)
        self.batch_predictions = self.make_batch_predictions_from_current_labels()
        self.batch_stats["predictions"] = dict(self.batch_predictions)
        self.batch_stats["done"] = int(len(self.batch_results))
        self.pages["batch"]["progress_bar"].setValue(idx)
        self.add_or_update_batch_list_item(key)
        self.update_batch_stats_label()
        self.update_buttons(self.engine is not None)
        if idx % BATCH_AUDIT_AUTOSAVE_INTERVAL == 0 or idx >= total:
            self.save_batch_audit_tmp()

        overlay = self.batch_mix_attention_rgb(self.batch_results[key])
        if overlay is None and isinstance(maps, dict) and "mix" in maps:
            overlay = heatmap_overlay(rgb, maps["mix"])
        self.show_left(rgb, "batch")
        if overlay is not None:
            self.show_right(overlay, "batch")
            self.last_attention_overlays = {"mix": np.ascontiguousarray(overlay.copy())}
        display_result = dict(result)
        display_result["source_type"] = "batch"
        self.update_status(display_result)
        self.select_batch_key(key)

        self.pages["batch"]["summary"].append(
            f"\n批量进度：{idx}/{total}\n当前文件：{key}\n当前判定：{result.get('result_cn', '')} | 导出值={self.batch_predictions[key]}"
        )

    def on_batch_error(self, msg: str):
        logger.error("批量预测错误：%s", msg)
        self.batch_stats["failed"] = int(self.batch_stats.get("failed", 0)) + 1
        self.update_batch_stats_label()
        self.pages["batch"]["summary"].append("\n批量错误：" + msg)

    def on_batch_finished(self, stats: Dict[str, Any]):
        self.batch_busy = False
        self.batch_paused = False
        self.batch_worker = None
        self.batch_stats.update(dict(stats))
        self.batch_predictions = self.make_batch_predictions_from_current_labels()
        self.batch_stats["predictions"] = dict(self.batch_predictions)
        self.batch_stats["done"] = len(self.batch_results)
        self.update_batch_stats_label()
        self.update_buttons(self.engine is not None)
        self.save_batch_audit_tmp()
        final_stats = self.recompute_batch_stats()
        acc = self.compute_batch_accuracy()
        status = "已停止" if stats.get("stopped") else "检测完成"
        acc_line = ""
        if acc.get("compared", 0) > 0:
            acc_line = f"\n对比准确率={acc['accuracy']:.2%} ({acc['correct']}/{acc['compared']})"
        self.pages["batch"]["summary"].append(
            f"\n\n批量{status}。"
            f"\n总数={final_stats['total']}，成功={final_stats['done']}，失败={final_stats['failed']}"
            f"\n有火={final_stats['fire']}，无火={final_stats['no_fire']}，疑似={final_stats['suspect']}，人工修改={final_stats['manual_changed']}，缓存读取={int(self.batch_stats.get('cached', 0))}"
            f"{acc_line}"
            f"\n临时 attention/meta 目录：{self.batch_stats.get('tmp_dir', '')}"
            "\n可以点击“导出 JSON”保存 {\"xxx.jpg\": 1/0}，人工修正会覆盖模型原始判断。"
        )

    def pause_batch_detection(self):
        if self.batch_worker is not None and self.batch_busy:
            self.batch_worker.pause()
            self.batch_paused = True
            self.pages["batch"]["summary"].append("\n已暂停批量检测。点击“继续批量”后会从下一张继续。")
        self.update_buttons(self.engine is not None)

    def resume_batch_detection(self):
        if self.batch_worker is not None and self.batch_busy:
            self.batch_worker.resume()
            self.batch_paused = False
            self.pages["batch"]["summary"].append("\n已继续批量检测。")
        self.update_buttons(self.engine is not None)

    def stop_batch_detection(self):
        if self.batch_worker is not None:
            self.batch_worker.stop()
            self.batch_paused = False
            self.pages["batch"]["summary"].append("\n已请求停止批量检测，当前图片处理完成后会停止。")
        self.update_buttons(self.engine is not None)

    def current_batch_key(self) -> str | None:
        if 0 <= self.batch_current_index < len(self.batch_order):
            return self.batch_order[self.batch_current_index]
        return None

    def on_batch_item_clicked(self, item: QtWidgets.QListWidgetItem):
        key = item.data(QtCore.Qt.UserRole) or item.text().replace("  [人工]", "")
        self.select_batch_key(str(key))

    def show_batch_prev(self):
        if not self.batch_order:
            return
        if self.batch_current_index <= 0:
            self.batch_current_index = 0
        else:
            self.batch_current_index -= 1
        self.select_batch_key(self.batch_order[self.batch_current_index])

    def show_batch_next(self):
        if not self.batch_order:
            return
        if self.batch_current_index < 0:
            self.batch_current_index = 0
        else:
            self.batch_current_index = min(len(self.batch_order) - 1, self.batch_current_index + 1)
        self.select_batch_key(self.batch_order[self.batch_current_index])

    def display_batch_item(self, key: str):
        item = self.batch_results.get(key)
        if not item:
            return
        rgb = item.get("rgb")
        maps = item.get("maps", {})
        if rgb is None:
            try:
                rgb = prepare_rgb_for_inference(imread_rgb(item["path"]), **self.get_resize_settings())[0]
            except Exception:
                return
        overlay = None
        if isinstance(maps, dict) and "mix" in maps:
            overlay = heatmap_overlay(rgb, maps["mix"])
        else:
            overlay_path = item.get("artifacts", {}).get("overlay_paths", {}).get("mix")
            if overlay_path and Path(overlay_path).exists():
                overlay = imread_rgb(overlay_path)
        self.show_left(rgb, "batch")
        if overlay is not None:
            self.show_right(overlay, "batch")
            self.last_attention_overlays = {"mix": np.ascontiguousarray(overlay.copy())}
        result = dict(item.get("result", {}))
        result["source_type"] = "batch"
        final_label = self.batch_final_label(item)
        result["result"] = "fire" if final_label == 1 else "no_fire"
        result["result_cn"] = "有火" if final_label == 1 else "无火"
        self.set_result_banner("batch", result)
        self.update_model_table_predictions(result)

        manual = item.get("manual_label")
        manual_text = "未修改" if manual is None else ("人工有火" if int(manual) == 1 else "人工无火")
        artifacts = item.get("artifacts", {})
        model_result = item.get("result", {})
        model_prob = result_prob(model_result, None)
        model_threshold = result_threshold(model_result, None)
        model_prob_text = "--" if model_prob is None else f"{model_prob:.4f}"
        model_threshold_text = "--" if model_threshold is None else f"{model_threshold:.3f}"
        lines = [
            f"当前：{key}",
            f"路径：{item.get('path', '')}",
            f"模型导出值：{item.get('pred_label', 0)} | 当前导出值：{self.batch_final_label(item)} | 人工审计：{manual_text}",
            f"模型结果：{model_result.get('result_cn', '')} | prob={model_prob_text} | 阈值={model_threshold_text}",
            f"meta：{artifacts.get('meta_path', '')}",
        ]
        overlay_paths = artifacts.get("overlay_paths", {})
        if overlay_paths:
            lines.append(f"Mix attention：{overlay_paths.get('mix', '')}")
        self.pages["batch"]["summary"].setPlainText("\n".join(lines))

    def update_current_batch_from_inference(self, result: Dict[str, Any], maps: Dict[str, np.ndarray], rgb: np.ndarray):
        key = result.get("batch_key") or self.current_batch_key()
        if key is None or key not in self.batch_results:
            return
        item = self.batch_results[key]
        item["result"] = dict(result)
        item["maps"] = {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()}
        item["rgb"] = np.ascontiguousarray(rgb.copy())
        item["pred_label"] = int(BatchInferenceRunnable._binary_export_label(result))
        item["artifacts"] = {}
        self.batch_predictions = self.make_batch_predictions_from_current_labels()
        self.batch_stats["predictions"] = dict(self.batch_predictions)
        self.add_or_update_batch_list_item(key)
        self.update_batch_stats_label()
        self.save_batch_audit_tmp()
        if self.audit_window is not None and self.audit_window.isVisible():
            self.audit_window.refresh_from_parent()

    def set_batch_manual_label(self, label):
        key = self.current_batch_key()
        if key is None or key not in self.batch_results:
            return
        self.batch_results[key]["manual_label"] = None if label is None else int(label)
        self.batch_predictions = self.make_batch_predictions_from_current_labels()
        self.batch_stats["predictions"] = dict(self.batch_predictions)
        self.add_or_update_batch_list_item(key)
        self.select_batch_key(key)
        self.update_batch_stats_label()
        self.save_batch_audit_tmp()
        if self.audit_window is not None and self.audit_window.isVisible():
            self.audit_window.refresh_from_parent()
        self.update_buttons(self.engine is not None)

    def export_batch_json(self):
        self.batch_predictions = self.make_batch_predictions_from_current_labels()
        if not self.batch_predictions:
            QtWidgets.QMessageBox.information(self, "没有批量结果", "请先完成一次文件夹批量检测。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出批量检测 JSON", "fire_batch_predictions.json", "JSON Files (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.batch_predictions, f, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(self, "导出完成", f"已导出：\n{path}")

    @staticmethod
    def unique_copy_destination(base_dir: Path, relative_key: str, source_path: str) -> Path:
        rel = Path(str(relative_key).replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            rel = Path(safe_artifact_stem(Path(source_path).name or str(relative_key)))
        if not rel.suffix:
            src_suffix = Path(source_path).suffix
            rel = rel.with_suffix(src_suffix or ".jpg")
        dest = base_dir / rel
        if not dest.exists():
            return dest
        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent
        for i in range(1, 10000):
            candidate = parent / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate
        digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
        return parent / f"{stem}_{digest}{suffix}"

    def _collect_batch_wrong_image_items(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Collect images whose model prediction disagrees with an audit label.

        Priority of reference labels:
        1) manual labels set in the audit window;
        2) imported ground-truth JSON labels.

        The comparison intentionally uses the raw model label (pred_label), not the
        final/manual-overridden label, so manually corrected mistakes can still be
        exported as model mistakes.
        """
        wrong_items: List[Dict[str, Any]] = []
        stats = {
            "total": 0,
            "comparable": 0,
            "unlabeled": 0,
            "wrong": 0,
            "manual_reference": 0,
            "ground_truth_reference": 0,
            "false_positive": 0,
            "false_negative": 0,
            "other_wrong": 0,
        }
        for key in self.batch_order:
            item = self.batch_results.get(key)
            if not item:
                continue
            stats["total"] += 1
            manual_label = item.get("manual_label")
            if manual_label is not None:
                reference_label = int(manual_label)
                reference_source = "manual_audit"
                stats["manual_reference"] += 1
            else:
                truth = ground_truth_label_for_key(self.batch_ground_truth or {}, key, item.get("path", ""))
                if truth is None:
                    stats["unlabeled"] += 1
                    continue
                reference_label = int(truth)
                reference_source = "ground_truth_json"
                stats["ground_truth_reference"] += 1
            stats["comparable"] += 1
            model_label = int(item.get("pred_label", 0))
            if model_label == reference_label:
                continue
            if model_label == 1 and reference_label == 0:
                error_type = "false_positive"
                stats["false_positive"] += 1
            elif model_label == 0 and reference_label == 1:
                error_type = "false_negative"
                stats["false_negative"] += 1
            else:
                error_type = "other_wrong"
                stats["other_wrong"] += 1
            stats["wrong"] += 1
            wrong_items.append({
                "key": key,
                "item": item,
                "model_label": model_label,
                "reference_label": reference_label,
                "reference_source": reference_source,
                "error_type": error_type,
            })
        return wrong_items, stats

    def export_batch_wrong_images(self):
        if not self.batch_results:
            QtWidgets.QMessageBox.information(self, "没有批量结果", "请先完成一次文件夹批量检测。")
            return
        wrong_items, stats = self._collect_batch_wrong_image_items()
        if stats.get("comparable", 0) <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "没有可对比标签",
                "请先在审计窗口给图片打人工标签，或在批量检测页导入对比 JSON。\n"
                "导出错误图片会用这些标签和模型原始判断进行比较。",
            )
            return
        if not wrong_items:
            QtWidgets.QMessageBox.information(
                self,
                "没有判断错误图片",
                f"当前可对比 {stats['comparable']} 张，未发现模型判断错误。\n"
                f"人工审计标签 {stats['manual_reference']} 张，对比 JSON 标签 {stats['ground_truth_reference']} 张。",
            )
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择判断错误图片的导出文件夹")
        if not folder:
            return
        out_root = Path(folder)
        fp_dir = out_root / "模型误报_预测有火_实际无火"
        fn_dir = out_root / "模型漏报_预测无火_实际有火"
        other_dir = out_root / "其他错误"
        for d in [fp_dir, fn_dir, other_dir]:
            d.mkdir(parents=True, exist_ok=True)

        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_dir": self.batch_stats.get("root_dir", ""),
            "ground_truth_path": self.batch_ground_truth_path,
            "note": "错误判断按模型原始 pred_label 与人工审计标签/对比 JSON 标签比较；不是按人工修正后的 final_label 比较。",
            "stats": dict(stats),
            "items": [],
            "errors": [],
        }
        copied = {"false_positive": 0, "false_negative": 0, "other_wrong": 0, "missing_or_failed": 0}
        for entry in wrong_items:
            key = entry["key"]
            item = entry["item"]
            source_path = str(item.get("path", ""))
            error_type = str(entry["error_type"])
            target_base = fp_dir if error_type == "false_positive" else fn_dir if error_type == "false_negative" else other_dir
            record = {
                "key": key,
                "source_path": source_path,
                "model_label": int(entry["model_label"]),
                "model_label_cn": "有火" if int(entry["model_label"]) == 1 else "无火",
                "reference_label": int(entry["reference_label"]),
                "reference_label_cn": "有火" if int(entry["reference_label"]) == 1 else "无火",
                "reference_source": entry["reference_source"],
                "error_type": error_type,
                "manual_label": item.get("manual_label"),
                "final_label": self.batch_final_label(item),
                "prob_fire": result_prob(item.get("result", {}), None),
                "threshold": result_threshold(item.get("result", {}), None),
            }
            if not source_path or not Path(source_path).exists():
                copied["missing_or_failed"] += 1
                record["export_path"] = ""
                record["error"] = "source missing"
                manifest["errors"].append(record)
                manifest["items"].append(record)
                continue
            try:
                dest = self.unique_copy_destination(target_base, key, source_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest)
                copied[error_type if error_type in copied else "other_wrong"] += 1
                record["export_path"] = str(dest)
                manifest["items"].append(record)
            except Exception as e:
                copied["missing_or_failed"] += 1
                record["export_path"] = ""
                record["error"] = str(e)
                manifest["errors"].append(record)
                manifest["items"].append(record)
        manifest["export_stats"] = dict(copied)
        manifest_path = out_root / "wrong_images_export_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(
            self,
            "导出完成",
            f"已导出判断错误图片 {sum(copied[k] for k in ['false_positive', 'false_negative', 'other_wrong'])} 张。\n"
            f"误报 {copied['false_positive']} 张，漏报 {copied['false_negative']} 张，其他错误 {copied['other_wrong']} 张。\n"
            f"缺失/失败 {copied['missing_or_failed']} 张。\n"
            f"目标目录：\n{out_root}\n清单：\n{manifest_path}"
        )

    def export_batch_split_images(self):
        if not self.batch_results:
            QtWidgets.QMessageBox.information(self, "没有批量结果", "请先完成一次文件夹批量检测。")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出有火/无火图片的目标文件夹")
        if not folder:
            return
        out_root = Path(folder)
        fire_dir = out_root / "有火图片"
        no_fire_dir = out_root / "无火图片"
        fire_dir.mkdir(parents=True, exist_ok=True)
        no_fire_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_dir": self.batch_stats.get("root_dir", ""),
            "threshold_settings": self.get_model_settings(),
            "items": [],
        }
        copied = {"fire": 0, "no_fire": 0, "missing": 0}
        errors = []
        for key in self.batch_order:
            item = self.batch_results.get(key)
            if not item:
                continue
            source_path = str(item.get("path", ""))
            label = self.batch_final_label(item)
            target_base = fire_dir if label == 1 else no_fire_dir
            if not source_path or not Path(source_path).exists():
                copied["missing"] += 1
                errors.append({"key": key, "path": source_path, "error": "source missing"})
                continue
            dest = self.unique_copy_destination(target_base, key, source_path)
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest)
                copied["fire" if label == 1 else "no_fire"] += 1
                manifest["items"].append({
                    "key": key,
                    "source_path": source_path,
                    "export_path": str(dest),
                    "label": int(label),
                    "label_cn": "有火" if label == 1 else "无火",
                    "manual_label": item.get("manual_label"),
                    "prob_fire": result_prob(item.get("result", {}), None),
                    "threshold": result_threshold(item.get("result", {}), None),
                })
            except Exception as e:
                copied["missing"] += 1
                errors.append({"key": key, "path": source_path, "error": str(e)})
        manifest["stats"] = dict(copied)
        manifest["errors"] = errors
        manifest_path = out_root / "batch_split_export_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(
            self,
            "导出完成",
            f"已导出有火图片 {copied['fire']} 张，无火图片 {copied['no_fire']} 张。\n"
            f"缺失/失败 {copied['missing']} 张。\n目标目录：\n{out_root}\n清单：\n{manifest_path}"
        )

    def export_batch_meta_json(self):
        if not self.batch_results:
            QtWidgets.QMessageBox.information(self, "没有批量结果", "请先完成一次文件夹批量检测。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出批量检测 Meta JSON", "fire_batch_meta.json", "JSON Files (*.json)")
        if not path:
            return
        stats = self.recompute_batch_stats()
        accuracy = self.compute_batch_accuracy()
        audited = max(int(stats.get("audited", 0)), 0)
        consistency = None
        if audited > 0:
            consistency = float(stats.get("model_match_manual", 0)) / float(audited)
        payload = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_dir": self.batch_stats.get("root_dir", ""),
            "tmp_dir": self.batch_stats.get("tmp_dir", self.batch_tmp_dir),
            "stats": {
                **stats,
                "manual_audit_consistency": consistency,
                "manual_audit_accuracy": consistency,
                "ground_truth_accuracy": accuracy,
                "cached": int(self.batch_stats.get("cached", 0)),
                "batch_size": int(self.batch_stats.get("batch_size", 1)),
            },
            "predictions": self.make_batch_predictions_from_current_labels(),
            "items": {},
        }
        for key in self.batch_order:
            item = self.batch_results.get(key)
            if not item:
                continue
            result = item.get("result", {})
            payload["items"][key] = {
                "path": item.get("path", ""),
                "model_label": int(item.get("pred_label", 0)),
                "manual_label": item.get("manual_label"),
                "final_label": self.batch_final_label(item),
                "result": BatchInferenceRunnable._compact_result_for_meta(result),
                "artifacts": item.get("artifacts", {}),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(self, "导出完成", f"已导出：\n{path}")

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "", VIDEO_EXTS)
        if not path:
            return
        self.stop_stream()
        self.source_type = "video"
        self.source_path = path
        self.frame_index = 0
        self.tabs.setCurrentWidget(self.videoTab)
        self.pages["video"]["summary"].setPlainText("视频已打开。左侧按原视频帧率播放；右侧在每次推理完成后更新 attention map。")
        self.start_frame_reader(path, "video")

    def open_camera(self):
        self.stop_stream()
        cam_id = int(self.camera_id_spinBox.value())
        self.source_type = "camera"
        self.source_path = f"camera_{cam_id}"
        self.frame_index = 0
        self.tabs.setCurrentWidget(self.cameraTab)
        self.pages["camera"]["summary"].setPlainText("摄像头正在打开。左侧显示原始采集帧；右侧在每次推理完成后更新 attention map。")
        self.start_frame_reader(cam_id, "camera")

    def start_frame_reader(self, source, source_type: str):
        self.frame_thread = FrameReaderThread(source, source_type, self)
        self.frame_thread.frame_ready.connect(self.on_frame_ready)
        self.frame_thread.error.connect(self.on_frame_error)
        self.frame_thread.finished_reading.connect(self.on_frame_reader_finished)
        if self.pages[source_type].get("stop_btn") is not None:
            self.pages[source_type]["stop_btn"].setEnabled(True)
        self.frame_thread.start()

    def on_frame_ready(self, rgb: np.ndarray, frame_index: int):
        self.frame_index = int(frame_index)
        source_type = self.source_type if self.source_type in ["video", "camera"] else "video"
        self.show_left(rgb, source_type)
        if not self.inference_busy:
            meta = {
                "source_type": source_type,
                "source_path": self.source_path,
                "frame_index": self.frame_index,
            }
            self.submit_inference(rgb, meta)

    def on_frame_error(self, msg: str):
        logger.error("视频源错误：%s", msg)
        source_type = self.source_type if self.source_type in self.pages else "video"
        if self.pages[source_type].get("stop_btn") is not None:
            self.pages[source_type]["stop_btn"].setEnabled(False)
        self.pages[source_type]["summary"].setPlainText("视频源打开失败：\n" + msg)

    def on_frame_reader_finished(self):
        source_type = self.source_type if self.source_type in self.pages else "video"
        if self.pages[source_type].get("stop_btn") is not None:
            self.pages[source_type]["stop_btn"].setEnabled(False)
        self.frame_thread = None

    def submit_inference(self, rgb: np.ndarray, meta: Dict[str, Any]):
        meta = dict(meta)
        source_type = meta.get("source_type", "image")
        is_mold_internal = bool(source_type == "mold" and meta.get("_mold_internal", False))
        if self.engine is None or self.inference_busy or self.batch_busy or (self.mold_busy and not is_mold_internal):
            return
        model_settings = self.get_model_settings()
        if not any(s.get("enabled", True) for s in model_settings):
            QtWidgets.QMessageBox.warning(self, "未加载模型", "请先加载模型。")
            return
        infer_rgb, valid_mask_np, resize_meta = prepare_rgb_for_inference(rgb, **self.get_resize_settings())
        meta["original_width"] = int(rgb.shape[1])
        meta["original_height"] = int(rgb.shape[0])
        meta["infer_width"] = int(infer_rgb.shape[1])
        meta["infer_height"] = int(infer_rgb.shape[0])
        meta.update(resize_meta)
        self.inference_busy = True
        self.pages[source_type]["summary"].setPlainText("正在预测... 请看上方结果标签、概率和专家路由信息。")
        worker = InferenceRunnable(
            self.engine,
            infer_rgb,
            model_settings,
            meta,
            valid_mask_np,
        )
        worker.signals.result.connect(self.on_inference_result)
        worker.signals.error.connect(self.on_inference_error)
        self.threadpool.start(worker)

    def on_inference_result(self, result: Dict[str, Any], maps: Dict[str, np.ndarray], rgb: np.ndarray):
        self.inference_busy = False
        source_type = result.get("source_type", "image")
        overlay = heatmap_overlay(rgb, maps["mix"])
        self.show_right(overlay, source_type)
        if source_type == "image":
            self.update_image_attention_grid(rgb, maps)
        else:
            self.last_attention_overlays = {
                key: np.ascontiguousarray(heatmap_overlay(rgb, attn).copy())
                for key, attn in maps.items()
            }
        if source_type == "batch":
            self.update_current_batch_from_inference(result, maps, rgb)
        self.update_status(result)

        record = {
            "result": result,
            "maps": {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()},
            "rgb": np.ascontiguousarray(rgb.copy()),
            "overlay": np.ascontiguousarray(overlay.copy()),
        }
        self.records.append(record)
        if source_type == "mold":
            self.handle_mold_inference_result(result, maps, rgb, overlay)

    def on_inference_error(self, msg: str):
        logger.error("预测任务失败：%s", msg)
        self.inference_busy = False
        source_type = self.source_type if self.source_type in self.pages else "image"
        self.pages[source_type]["summary"].setPlainText("识别出错：\n" + msg)
        self.set_result_banner(source_type, None, error_msg="识别出错")
        if source_type == "mold" and self.mold_busy:
            self.finish_mold_scan(error_msg=msg)

    def set_result_banner(self, source_type: str, result=None, error_msg=None):
        page = self.pages[source_type]
        result_label = page["result_label"]
        prob_label = page["prob_label"]
        if error_msg:
            result_label.setText("⚠️ 识别出错")
            result_label.setStyleSheet("font-size:26px; font-weight:700; border-radius:8px; background:#FFF7ED; color:#C2410C;")
            prob_label.setText(error_msg)
            return
        if result is None:
            result_label.setText("等待预测")
            result_label.setStyleSheet("font-size:26px; font-weight:700; border-radius:8px; background:#EEF2F7; color:#334155;")
            prob_label.setText("概率：-- | 阈值：--")
            return
        result_key = str(result.get("result", "no_fire"))
        if result_key == "fire":
            result_label.setText("🔥 有火")
            result_label.setStyleSheet("font-size:30px; font-weight:800; border-radius:8px; background:#FEE2E2; color:#B91C1C;")
        elif result_key == "suspect":
            result_label.setText("⚠️ 疑似")
            result_label.setStyleSheet("font-size:30px; font-weight:800; border-radius:8px; background:#FEF3C7; color:#92400E;")
        else:
            result_label.setText("✅ 无火")
            result_label.setStyleSheet("font-size:30px; font-weight:800; border-radius:8px; background:#DCFCE7; color:#166534;")
        prob_value = result_prob(result, None)
        threshold_value = result_threshold(result, None)
        prob_text = "--" if prob_value is None else f"{prob_value:.4f}"
        threshold_text = "--" if threshold_value is None else f"{threshold_value:.3f}"
        prob_label.setText(
            f"prob_fire={prob_text} | "
            f"threshold={threshold_text} | "
            f"{result.get('threshold_source', 'model')}"
        )

    def update_status(self, r: Dict[str, Any]):
        source_type = r.get("source_type", "image")
        self.set_result_banner(source_type, r)
        self.update_model_table_predictions(r)

        expert_data = sorted(r.get("expert_data", []), key=lambda item: float(item.get("posterior_weight", item.get("weight", 0.0))), reverse=True)
        expert_lines = [
            f"  {item.get('expert_name', '')}: prior={item.get('prior_weight', 0.0):.3f}, "
            f"post={item.get('posterior_weight', item.get('weight', 0.0)):.3f}, "
            f"conf={item.get('confidence', 0.0):.3f}, prob={item.get('prob_fire', 0.0):.4f}"
            for item in expert_data[:8]
        ]
        prob_value = result_prob(r, None)
        threshold_value = result_threshold(r, None)
        prob_text = "--" if prob_value is None else f"{prob_value:.4f}"
        threshold_text = "--" if threshold_value is None else f"{threshold_value:.3f}"
        lines = [
            f"来源：{self.SOURCE_CN.get(source_type, source_type)} | frame={r.get('frame_index', 0)} | "
            f"infer={r.get('infer_width', 0)}×{r.get('infer_height', 0)} | original={r.get('original_width', 0)}×{r.get('original_height', 0)}",
            f"结果：{r.get('result_cn', '')} | prob_fire={prob_text} | threshold={threshold_text} | "
            f"阈值来源={r.get('threshold_source', '')}",
            f"模型：{r.get('model_name', '')}",
            f"Prompt：正 {r.get('owl_positive_prompt_count', 0)} | 负 {r.get('owl_negative_prompt_count', 0)} | "
            f"疑 {r.get('owl_suspicious_prompt_count', 0)}",
            "",
            f"分支概率：global={r.get('global_probability', 0.0):.4f}, "
            f"local={r.get('local_probability', 0.0):.4f}, semantic={r.get('semantic_probability', 0.0):.4f}",
            f"分支置信度：global={r.get('global_confidence', 0.0):.3f}, "
            f"local={r.get('local_confidence', 0.0):.3f}, semantic={r.get('semantic_confidence', 0.0):.3f}",
            f"裁决权重：global={r.get('branch_w_global', 0.0):.3f}, local={r.get('branch_w_local', 0.0):.3f}",
            f"OWL三组语义：positive={r.get('semantic_positive', 0.0):+.3f}, "
            f"negative={r.get('semantic_negative', 0.0):+.3f}, suspicious={r.get('semantic_suspicious', 0.0):+.3f}",
            f"Router prior→posterior Top-1 是否变化：{r.get('router_change', 0)}",
            "",
            "双向专家路由 Top：",
            *(expert_lines if expert_lines else ["  --"]),
            f"当前会话已缓存识别结果：{len(self.records) + 1} 条",
        ]
        self.pages[source_type]["summary"].setPlainText("\n".join(lines))
        if source_type == "image":
            self.update_tile_table(r)

    def set_model_row_color(self, row: int, pred):
        if pred is None:
            bg = QtGui.QColor("#FFFFFF")
            fg = QtGui.QColor("#111827")
            spin_style = "background:#FFFFFF; color:#111827;"
        elif pred == "disabled":
            bg = QtGui.QColor("#E5E7EB")
            fg = QtGui.QColor("#374151")
            spin_style = "background:#E5E7EB; color:#374151;"
        elif int(pred) == 1:
            bg = QtGui.QColor("#B91C1C")
            fg = QtGui.QColor("#FFFFFF")
            spin_style = "background:#B91C1C; color:#FFFFFF; font-weight:700;"
        else:
            bg = QtGui.QColor("#064E3B")
            fg = QtGui.QColor("#FFFFFF")
            spin_style = "background:#064E3B; color:#FFFFFF; font-weight:700;"
        for col in range(self.modelTable.columnCount()):
            item = self.modelTable.item(row, col)
            if item is not None:
                item.setBackground(bg)
                item.setForeground(fg)
        spin = self.modelTable.cellWidget(row, 3)
        if spin is not None:
            spin.setStyleSheet(spin_style)

    def update_model_table_predictions(self, r: Dict[str, Any]):
        self.latest_model_prediction = int(r.get("pred", 0))
        prob_value = result_prob(r, None)
        threshold_value = result_threshold(r, None)
        prob_text = "--" if prob_value is None else f"{prob_value:.4f}"
        threshold_text = "--" if threshold_value is None else f"{threshold_value:.3f}"
        self.latestProbLabel.setText(
            f"概率：{prob_text} | 阈值：{threshold_text}"
        )
        self.save_gui_config()

    def update_tile_table(self, r: Dict[str, Any]):
        table = self.pages["image"].get("tile_table")
        if table is None:
            return
        expert_data = sorted(r.get("expert_data", []), key=lambda x: float(x.get("weight", 0.0)), reverse=True)
        table.setRowCount(len(expert_data))
        for row, item in enumerate(expert_data):
            values = [
                str(item.get("expert_name", "")),
                f"{item.get('weight', 0.0):.4f}",
                f"{item.get('prob_fire', 0.0):.4f}",
            ]
            for col, val in enumerate(values):
                cell = QtWidgets.QTableWidgetItem(val)
                cell.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(row, col, cell)
        table.resizeColumnsToContents()

    def show_left(self, rgb: np.ndarray, source_type: str):
        self.last_left[source_type] = rgb
        label = self.pages[source_type]["left"]
        label.setPixmap(rgb_to_qpixmap(rgb, label))

    def show_right(self, rgb: np.ndarray, source_type: str):
        self.last_right[source_type] = rgb
        label = self.pages[source_type]["right"]
        label.setPixmap(rgb_to_qpixmap(rgb, label))

    def clear_attention_display(self, source_type: str):
        self.last_right[source_type] = None
        self.last_attention_overlays = {}
        page = self.pages.get(source_type, {})
        right_label = page.get("right")
        if right_label is not None:
            right_label.clear()
            right_label.setText("正在重新生成 attention map...")
        if source_type == "image":
            self.last_image_maps = {}
            for label in page.get("map_labels", {}).values():
                label.clear()
                label.setText("正在重新生成")

    def update_image_attention_grid(self, rgb: np.ndarray, maps: Dict[str, np.ndarray]):
        labels = self.pages.get("image", {}).get("map_labels", {})
        if not labels:
            return
        self.last_image_rgb = np.ascontiguousarray(rgb.copy())
        self.last_image_maps = {k: np.asarray(v, dtype=np.float32) for k, v in maps.items()}
        self.last_attention_overlays = {}
        for key, label in labels.items():
            if key not in maps:
                continue
            img = heatmap_overlay(rgb, maps[key])
            self.last_attention_overlays[key] = np.ascontiguousarray(img.copy())
            label.setPixmap(rgb_to_qpixmap(img, label, smooth=True))

    def open_attention_preview(self, key: str):
        key = str(key)
        if key not in self.last_attention_overlays:
            QtWidgets.QMessageBox.information(self, "没有 attention 图", "请先完成一次图片识别或批量检测预览。")
            return
        titles = {
            "global": "Global attention map",
            "local": "Local attention map",
            "semantic": "Semantic attention map",
            "mix": "Mix attention map",
        }
        dialog = ImagePreviewDialog(titles.get(key, key), self.last_attention_overlays[key], self)
        dialog.exec_()

    def open_batch_current_external(self, *_args):
        key = self.current_batch_key()
        if key is None or key not in self.batch_results:
            QtWidgets.QMessageBox.information(self, "没有当前图片", "请先在批量检测结果中选择一张图片。")
            return
        item = self.batch_results[key]
        try:
            raw_rgb = imread_rgb(item["path"])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "无法打开图片", str(e))
            return

        include_attention = bool(self.pages["batch"]["include_attention_check"].isChecked())
        view_rgb = raw_rgb
        if include_attention:
            overlay = None
            overlay_path = item.get("artifacts", {}).get("overlay_paths", {}).get("mix")
            if overlay_path and Path(overlay_path).exists():
                try:
                    overlay = imread_rgb(overlay_path)
                except Exception:
                    overlay = None
            if overlay is None and "mix" in item.get("maps", {}):
                infer_rgb = prepare_rgb_for_inference(raw_rgb, **self.get_resize_settings())[0]
                overlay = heatmap_overlay(infer_rgb, item["maps"]["mix"])
            if overlay is not None:
                view_rgb = side_by_side_rgb(raw_rgb, overlay)

        fd, path = tempfile.mkstemp(prefix="fire_batch_preview_", suffix=".png")
        os.close(fd)
        imwrite_png(path, view_rgb)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def stop_stream(self):
        if self.frame_thread is not None:
            self.frame_thread.stop()
            self.frame_thread.wait(1500)
            self.frame_thread = None
        for source_type in ["video", "camera"]:
            page = self.pages.get(source_type, {})
            if page.get("stop_btn") is not None:
                page["stop_btn"].setEnabled(False)

    def export_results(self):
        if not self.records:
            QtWidgets.QMessageBox.information(self, "没有可导出的结果", "当前会话还没有完成任何一次识别。")
            return
        base_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not base_dir:
            return

        export_dir = Path(base_dir) / time.strftime("attention_export_%Y%m%d_%H%M%S")
        export_dir.mkdir(parents=True, exist_ok=True)
        maps_dir = export_dir / "attention_maps"
        overlays_dir = export_dir / "overlays"
        raw_dir = export_dir / "raw_frames"
        maps_dir.mkdir(exist_ok=True)
        overlays_dir.mkdir(exist_ok=True)
        raw_dir.mkdir(exist_ok=True)

        csv_rows = []
        json_rows = []
        map_keys = ["global", "local", "semantic", "mix"]

        for idx, rec in enumerate(self.records):
            result = dict(rec["result"])
            prefix = f"{idx:05d}_{result.get('source_type', 'src')}_frame{result.get('frame_index', 0)}"

            imwrite_png(raw_dir / f"{prefix}_raw.png", rec["rgb"])
            imwrite_png(overlays_dir / f"{prefix}_mix_overlay.png", rec["overlay"])
            for key in map_keys:
                if key in rec["maps"]:
                    imwrite_png(maps_dir / f"{prefix}_{key}.png", map_to_gray(rec["maps"][key]))

            flat = {k: v for k, v in result.items() if k not in ["tile_data", "tile_boxes", "model_results", "model_settings"]}
            flat["tile_data_json"] = json.dumps(result.get("tile_data", []), ensure_ascii=False)
            flat["model_results_json"] = json.dumps(result.get("model_results", []), ensure_ascii=False)
            flat["model_settings_json"] = json.dumps(result.get("model_settings", []), ensure_ascii=False)
            csv_rows.append(flat)
            json_rows.append(result)

        csv_path = export_dir / "predictions.csv"
        fieldnames = sorted({k for row in csv_rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        with open(export_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(json_rows, f, ensure_ascii=False, indent=2)

        QtWidgets.QMessageBox.information(self, "导出完成", f"已导出到：\n{export_dir}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for source_type in ["image", "mold", "batch", "video", "camera"]:
            if self.last_left.get(source_type) is not None:
                label = self.pages[source_type]["left"]
                label.setPixmap(rgb_to_qpixmap(self.last_left[source_type], label))
            if self.last_right.get(source_type) is not None:
                label = self.pages[source_type]["right"]
                label.setPixmap(rgb_to_qpixmap(self.last_right[source_type], label))
        labels = self.pages.get("image", {}).get("map_labels", {})
        for key, label in labels.items():
            if key in self.last_attention_overlays:
                label.setPixmap(rgb_to_qpixmap(self.last_attention_overlays[key], label, smooth=True))

    def closeEvent(self, event):
        if self.model_loading and self.model_load_thread is not None and self.model_load_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "模型正在加载",
                "模型加载尚未完成。为避免后台加载线程被强制销毁，请等待加载结束后再关闭程序。",
            )
            event.ignore()
            return
        self.save_gui_config()
        self.stop_stream()
        event.accept()


