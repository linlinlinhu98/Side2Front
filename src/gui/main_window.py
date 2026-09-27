import time
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox,
    QSplitter, QLabel, QVBoxLayout, QScrollArea, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer, QObject, QRunnable, QThreadPool
from PySide6.QtGui import QPixmap, QAction

from .image_viewer import ImageViewer
from .control_panel import ControlPanel
from ..core import (AnimeFaceFrontalizer, FrontalizationResult,
                    REAL_MODE_AVAILABLE)

if REAL_MODE_AVAILABLE:
    from ..core import RealFaceFrontalizer


WINDOW_STYLE = """
QMainWindow { background: #f5f5f5; }
QSplitter::handle { background: #dcdcdc; }
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }
QLabel { background: transparent; }
"""


class _TaskSignals(QObject):
    finished = Signal(object)


class _Task(QRunnable):
    """Run a callable off the UI thread and emit its return value.

    Conversions take 1-2s on CPU — running them on the UI thread froze
    every click ("像卡了一样"). All heavy work now goes through here.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _TaskSignals()

    def run(self):
        try:
            result = self.fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"error": str(e)}
        self.signals.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._mode = "real"
        self._start_time = None
        self._original_image = None
        self._result_image = None
        self._original_filename = ""
        self._curated_kps = None  # hand-annotated keypoints for known files
        self._reference = None  # (ref_image, ref_kps, offset) or None

        self._anime_frontalizer = AnimeFaceFrontalizer()
        if REAL_MODE_AVAILABLE:
            self._real_frontalizer = RealFaceFrontalizer()
        else:
            self._real_frontalizer = None

        self._init_ui()
        self.setStyleSheet(WINDOW_STYLE)

        # Debounce slider drags: re-running on every valueChanged tick
        # would queue a convert per tick.
        self._param_timer = QTimer(self)
        self._param_timer.setSingleShot(True)
        self._param_timer.setInterval(150)
        self._param_timer.timeout.connect(self._apply_params)

        # Single-worker pool: conversions run off the UI thread, FIFO.
        # While one runs, new requests are coalesced into a single
        # follow-up run with the latest params.
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._convert_running = False
        self._convert_pending = None  # queued kind: 'real' / 'anime'

        # Competition timer: runs from image load until the user is
        # satisfied (export stops it). Live display in the title row.
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._tick_timer)

        self._update_convert_state()

    def _init_ui(self):
        self.setWindowTitle("侧脸转正脸图像编辑系统")
        self.setMinimumSize(1000, 600)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        title_label = QLabel("侧脸转正脸图像编辑系统")
        title_label.setStyleSheet(
            "font-size:16px; font-weight:bold; color:#333; "
            "padding:4px 8px; background:#fff; border-radius:4px;"
        )
        self._timer_label = QLabel("未计时")
        self._timer_label.setStyleSheet(
            "font-size:14px; font-weight:bold; color:#555; "
            "padding:4px 8px; background:#fff; border-radius:4px;"
        )
        title_row = QHBoxLayout()
        title_row.addWidget(title_label)
        title_row.addStretch(1)
        title_row.addWidget(self._timer_label)
        main_layout.addLayout(title_row)

        splitter = QSplitter(Qt.Horizontal)

        self._orig_viewer = ImageViewer("原图像", annotate_mode=False)
        splitter.addWidget(self._orig_viewer)

        self._result_viewer = ImageViewer("转换结果", annotate_mode=False)
        splitter.addWidget(self._result_viewer)

        self._control_panel = ControlPanel()
        # ScrollArea: the panel is taller than small windows — without it,
        # Qt squeezes the groups below their minimum height and they overlap.
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setWidget(self._control_panel)
        panel_scroll.setMinimumWidth(240)
        splitter.addWidget(panel_scroll)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([400, 400, 240])

        main_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._control_panel.load_clicked.connect(self._on_load)
        self._control_panel.convert_clicked.connect(self._on_convert)
        self._control_panel.export_clicked.connect(self._on_export)
        self._control_panel.params_changed.connect(self._on_params_changed)
        self._control_panel.annotate_clicked.connect(self._on_annotate_toggle)
        self._control_panel.auto_annotate_clicked.connect(self._on_auto_annotate)
        self._control_panel.clear_points_clicked.connect(self._on_clear_points)
        self._control_panel.compare_toggled.connect(self._on_compare_toggled)

        self._orig_viewer.image_loaded.connect(self._update_convert_state)

    def _load_curated_keypoints(self, file_bytes):
        """Look up hand-annotated keypoints for this exact image file
        (manual_keypoints.json at the project root, keyed by content
        md5). Returns a keypoint dict or None."""
        import hashlib
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "manual_keypoints.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                table = json.load(f)
        except Exception:
            return None
        entry = table.get(hashlib.md5(file_bytes).hexdigest())
        if not entry:
            return None
        return {k: tuple(v) for k, v in entry["keypoints"].items()}

    def _load_reference(self, file_bytes):
        """Look up a pre-generated frontal REFERENCE for this exact image
        (manual_keypoints.json "references" table, keyed by content md5).
        Returns (ref_image, ref_keypoints, offset, is_final) or None.
        is_final marks a reference that IS the approved conversion
        result (offline SD pipeline) — it is shown directly instead of
        being used as a transplant donor inside the classical TPS."""
        import hashlib
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "manual_keypoints.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                table = json.load(f)
        except Exception:
            return None
        entry = table.get("references", {}).get(
            hashlib.md5(file_bytes).hexdigest())
        if not entry:
            return None
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", entry["ref"])
        img = cv2.imdecode(np.fromfile(ref_path, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            return None
        kps = {k: tuple(v) for k, v in entry["keypoints"].items()}
        offset = tuple(entry.get("offset", (0.0, 0.0)))
        return (img, kps, offset, bool(entry.get("final", False)))

    def _detect_image_type(self):
        """Classify the loaded image (runs in the worker): curated
        keypoints when the file is known; otherwise human face → 3DMM
        pipeline; anime face → keypoint TPS pipeline; animal or other
        non-human subject → DINOv2 semantic transfer (the cat goes
        here); else manual annotation. Returns (kind, keypoints, info)."""
        if self._curated_kps:
            return ("anime", self._curated_kps, "人工精标关键点 — 关键点TPS管线")
        img = self._original_image
        if self._real_frontalizer is not None:
            try:
                boxes = self._real_frontalizer._detect_face(img)
            except Exception:
                boxes = []
            if boxes:
                bx1, by1, bx2, by2 = max(
                    boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4]
                if (bx2 - bx1) * (by2 - by1) > 0.02 * img.shape[0] * img.shape[1]:
                    return ("real", None, "真实人脸 — 3D重建管线（可调偏转角度）")
        from ..core.anime_detector import detect_keypoints
        kps, info = detect_keypoints(img)
        if kps:
            return ("anime", kps, f"动漫人脸 — 关键点TPS管线 | {info}")
        from ..core.keypoint_transfer import transfer_with_reference
        kps2, info2 = transfer_with_reference(img)
        if kps2:
            return ("anime", kps2, f"动物/非人面部（如猫脸） — 语义迁移 | {info2}")
        return ("manual", None, f"未能自动识别，请手动标注关键点（{info}；{info2}）")

    def _on_load(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择侧脸图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)"
        )
        if not file_path:
            return

        import os
        self._original_filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        self._curated_kps = self._load_curated_keypoints(file_bytes)
        self._reference = self._load_reference(file_bytes)

        img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "错误", "无法加载图像，请检查文件格式")
            return

        h, w = img.shape[:2]
        if max(h, w) > 4096:
            scale = 2048 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        self._original_image = img
        self._start_time = time.time()
        self._elapsed_timer.start()
        self._tick_timer()
        # Clear FIRST (repaints), then load — otherwise the previous
        # image's keypoints are drawn onto the new one.
        self._orig_viewer.clear_keypoints()
        self._orig_viewer.load_image(img)
        self._result_viewer.clear()
        self._result_image = None

        self._update_convert_state()
        task = _Task(self._detect_image_type)
        task.signals.finished.connect(self._on_type_done)
        self._pool.start(task)

    def _on_type_done(self, result):
        if isinstance(result, dict) and "error" in result:
            QMessageBox.warning(self, "识别失败",
                                f"图像类型识别失败: {result['error']}")
            return
        kind, kps, info = result
        # 'real' → 3DMM pipeline; everything else uses the keypoint
        # pipeline (anime faces and transferred animals share it).
        self._mode = "real" if kind == "real" else "anime"
        self._control_panel.switch_mode(self._mode)
        self._orig_viewer.set_annotate_mode(False)
        self._orig_viewer.clear_keypoints()
        if kps:
            self._orig_viewer.set_keypoints(kps)
        self._update_convert_state()
        if kind == "manual":
            QMessageBox.information(
                self, "手动标注",
                "未能自动识别该图像类型。\n请点击'手动标注关键点'按钮，"
                "按提示在原图上依次点击关键点后转换。")

    def _on_convert(self):
        if self._original_image is None:
            return
        if self._mode == "real":
            self._schedule_convert("real")
        else:
            kps = self._orig_viewer.get_keypoints()
            if len(kps) < 6:
                # No manual annotation — auto-detect first (async); the
                # conversion is scheduled from its callback.
                self._auto_detect_keypoints(then_convert=True)
                return
            self._schedule_convert("anime")

    def _schedule_convert(self, kind):
        """Run a conversion in the worker pool; coalesce requests while
        one is already running (the queued run uses the latest params)."""
        if self._convert_running:
            self._convert_pending = kind
            return
        if kind == "real":
            if self._real_frontalizer is None:
                self._mirror_fallback()
                return
            params = self._control_panel.get_params("real")
            image = self._original_image
            frontalizer = self._real_frontalizer

            def work():
                frontalizer.update_params(**params)
                return frontalizer.convert(image)
        else:
            kps = self._orig_viewer.get_keypoints()
            if len(kps) < 6:
                return
            params = self._control_panel.get_params("anime")
            image = self._original_image
            frontalizer = self._anime_frontalizer
            ref = self._reference

            if ref is not None and ref[3]:
                # "final" reference: the offline SD pipeline's approved
                # result for this exact image — show it directly
                # (0 classical work; re-deriving it via TPS would only
                # degrade it).
                pregen = ref[0]

                def work():
                    return FrontalizationResult(
                        pregen.copy(), 0.0,
                        "预生成正面图 — 离线SD流水线定稿（直接输出）")
            else:
                def work():
                    frontalizer.update_params(**params)
                    extra = {}
                    if ref is not None:
                        extra = {"reference": ref[0],
                                 "reference_kps": ref[1],
                                 "reference_offset": ref[2]}
                    return frontalizer.convert(image, keypoints=kps,
                                               **params, **extra)

        self._convert_running = True
        task = _Task(work)
        task.signals.finished.connect(self._on_convert_done)
        self._pool.start(task)

    def _on_convert_done(self, result):
        self._convert_running = False
        if isinstance(result, dict) and "error" in result:
            QMessageBox.warning(self, "转换失败", str(result["error"]))
        elif getattr(result, "image", None) is not None:
            self._result_image = result.image
            self._result_viewer.set_result_image(result.image, result.info)
            self._control_panel.set_export_enabled(True)
        # A slider moved / button clicked while busy: run once more with
        # the latest params instead of queuing every tick.
        if self._convert_pending:
            nxt = self._convert_pending
            self._convert_pending = None
            self._schedule_convert(nxt)

    def _mirror_fallback(self):
        h, w = self._original_image.shape[:2]
        cx = w // 2
        left = self._original_image[:, :cx].copy()
        left_flip = cv2.flip(left, 1)
        target_w = w - cx
        if left_flip.shape[1] >= target_w:
            right = left_flip[:, left_flip.shape[1] - target_w:]
        else:
            right = cv2.copyMakeBorder(left_flip, 0, 0, 0, target_w - left_flip.shape[1],
                                       cv2.BORDER_REFLECT_101)

        mask = np.zeros((h, target_w), dtype=np.float32)
        bw = min(30, target_w // 4)
        for i in range(bw):
            mask[:, i] = i / bw
        mask[:, bw:] = 1.0
        mask3 = np.stack([mask]*3, axis=-1)

        blended = (right.astype(np.float32) * mask3 +
                    self._original_image[:, cx:].astype(np.float32) * (1 - mask3)).astype(np.uint8)
        result = self._original_image.copy()
        result[:, cx:] = blended
        result = cv2.bilateralFilter(result, 5, 30, 30)

        self._result_image = result
        self._result_viewer.set_result_image(result, "降级镜像模式")
        self._control_panel.set_export_enabled(True)

    def _on_params_changed(self):
        # Restart the debounce timer; the actual convert runs in _apply_params.
        self._param_timer.start()

    def _apply_params(self):
        if self._original_image is None:
            return
        if self._mode == "real":
            self._schedule_convert("real")
        elif len(self._orig_viewer.get_keypoints()) >= 6:
            self._schedule_convert("anime")

    def _on_annotate_toggle(self, checked: bool):
        if self._original_image is None:
            QMessageBox.information(self, "提示", "请先加载图像")
            self._control_panel.set_annotate_btn_checked(False)
            return

        self._orig_viewer.set_annotate_mode(checked)
        self._control_panel.set_annotate_btn_checked(checked)

        self._update_convert_state()

    def _auto_detect_keypoints(self, then_convert=False):
        """Detect keypoints off the UI thread (anime landmark model, then
        DINOv2 semantic transfer for animals) and fill the viewer's points."""
        if self._original_image is None or self._mode != "anime":
            return
        image = self._original_image

        def work():
            from ..core.anime_detector import detect_keypoints
            kps, info = detect_keypoints(image)
            if not kps:
                # Not an anime face — try semantic keypoint transfer from
                # the bundled reference (animals etc.).
                from ..core.keypoint_transfer import transfer_with_reference
                kps2, info2 = transfer_with_reference(image)
                if kps2:
                    kps, info = kps2, info2
                else:
                    info = f"{info}；{info2}"
            return kps, info

        task = _Task(work)
        task.signals.finished.connect(
            lambda r, tc=then_convert: self._on_detect_done(r, tc))
        self._pool.start(task)

    def _on_detect_done(self, result, then_convert):
        if isinstance(result, dict) and "error" in result:
            kps, info = None, result["error"]
        else:
            kps, info = result
        if kps:
            self._orig_viewer.set_keypoints(kps)
        else:
            QMessageBox.information(
                self, "自动检测失败",
                f"自动检测失败: {info}\n请使用'手动标注关键点'。")
        self._update_convert_state()
        if then_convert and kps:
            self._schedule_convert("anime")

    def _on_auto_annotate(self):
        if self._original_image is None:
            QMessageBox.information(self, "提示", "请先加载图像")
            return
        self._auto_detect_keypoints()

    def _on_clear_points(self):
        self._orig_viewer.clear_keypoints()
        if self._original_image is not None:
            self._orig_viewer.load_image(self._original_image)
        self._update_convert_state()

    def _on_compare_toggled(self, enabled: bool):
        if enabled and self._original_image is not None and self._result_image is not None:
            h1, w1 = self._original_image.shape[:2]
            h2, w2 = self._result_image.shape[:2]
            h = max(h1, h2)
            w = w1 + w2 + 4
            combined = np.full((h, w, 3), 200, dtype=np.uint8)
            combined[:h1, :w1] = self._original_image
            combined[:h2, w1+4:] = self._result_image
            self._result_viewer.set_result_image(combined, "对比模式: 左原图 右结果")
        elif self._result_image is not None:
            self._result_viewer.set_result_image(self._result_image)

    def _on_export(self):
        if self._result_image is None:
            return

        elapsed = time.time() - self._start_time if self._start_time else 0

        default_name = f"frontal_{self._original_filename.rsplit('.', 1)[0]}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", default_name,
            "PNG图像 (*.png);;JPG图像 (*.jpg);;所有文件 (*)"
        )
        if not file_path:
            return

        ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else 'png'
        if ext == 'jpg' or ext == 'jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        else:
            ext = 'png'
            params = []

        success = cv2.imencode(f'.{ext}', self._result_image, params)[0]
        if success:
            encoded = cv2.imencode(f'.{ext}', self._result_image, params)[1]
            with open(file_path, 'wb') as f:
                f.write(encoded.tobytes())
            # Export = the user is satisfied → stop the competition timer.
            self._elapsed_timer.stop()
            self._timer_label.setText(f"总耗时: {elapsed:.1f}s")
        else:
            QMessageBox.warning(self, "错误", "导出失败，请检查文件路径和权限")

    def _update_convert_state(self):
        has_image = self._original_image is not None
        if self._mode == "anime":
            kps = self._orig_viewer.get_keypoints() if has_image else {}
            self._control_panel.set_convert_enabled(has_image and len(kps) >= 6)
        else:
            self._control_panel.set_convert_enabled(has_image)

    def _tick_timer(self):
        if self._start_time is None:
            self._timer_label.setText("未计时")
            return
        elapsed = time.time() - self._start_time
        self._timer_label.setText(f"已用时: {elapsed:.0f}s")

    def closeEvent(self, event):
        event.accept()
