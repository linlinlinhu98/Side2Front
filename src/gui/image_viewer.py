import numpy as np
import cv2
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush


KEYPOINT_TEMPLATE = [
    ("发际线左", "hairline_left"),
    ("发际线中", "hairline_center"),
    ("发际线右", "hairline_right"),
    ("左眉内", "brow_left_inner"),
    ("左眉外", "brow_left_outer"),
    ("左眼内", "eye_left_inner"),
    ("左眼外", "eye_left_outer"),
    ("鼻尖", "nose_tip"),
    ("鼻翼左", "nose_left"),
    ("嘴角左", "mouth_left"),
    ("嘴中心", "mouth_center"),
    ("嘴角右", "mouth_right"),
    ("下巴尖", "chin_tip"),
    ("下巴左", "chin_left"),
    ("下巴右", "chin_right"),
    ("耳上", "ear_top"),
    ("耳下", "ear_bottom"),
]


class ImageViewer(QWidget):
    image_loaded = Signal(bool)
    point_clicked = Signal(str, float, float)

    def __init__(self, title="图像", parent=None, annotate_mode=False):
        super().__init__(parent)
        self._title = title
        self._annotate_mode = annotate_mode
        self._image = None
        self._display_pixmap = None
        self._zoom = 1.0
        self._scale = 1.0  # fit-to-canvas base scale x user zoom, used for drawing/mapping
        self._pan_pos = None
        self._offset = QPointF(0, 0)

        self._keypoints = {}
        self._kp_idx = 0
        self._face_boxes = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = QLabel(self._title)
        self._header.setStyleSheet(
            "background:#e8e8e8; color:#666; font-size:12px; "
            "font-weight:600; padding:6px 10px; border-bottom:1px solid #ddd;"
        )
        layout.addWidget(self._header)

        self._canvas = QLabel("点击或拖入图像")
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setStyleSheet(
            "background:#f5f5f5; color:#aaa; font-size:13px; "
            "border:2px dashed #bbb; margin:8px;"
        )
        self._canvas.setMinimumSize(300, 300)
        layout.addWidget(self._canvas, 1)

        self.setAcceptDrops(True)

    def set_annotate_mode(self, enabled: bool):
        self._annotate_mode = enabled
        if enabled:
            self._kp_idx = 0
            self._keypoints = {}
        self._update_display()

    def get_keypoints(self) -> dict:
        return self._keypoints

    def clear_keypoints(self):
        self._keypoints = {}
        self._kp_idx = 0
        # NB: the drawing lives in the canvas pixmap — QWidget.update()
        # alone does NOT repaint it, so stale keypoints stayed visible
        # after switching images.
        self._update_display()

    def set_keypoints(self, keypoints: dict):
        """Fill keypoints programmatically (auto-detection)."""
        self._keypoints = {k: tuple(v) for k, v in keypoints.items()}
        self._kp_idx = len(KEYPOINT_TEMPLATE)
        self._update_display()

    def set_face_boxes(self, boxes):
        self._face_boxes = boxes
        self.update()

    def load_image(self, cv_image: np.ndarray):
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        self._image = cv_image
        self._zoom = 1.0  # refit each newly loaded image to the canvas
        self._offset = QPointF(0, 0)
        self._update_display()
        self.image_loaded.emit(True)

    def get_image(self) -> np.ndarray:
        return self._image

    def set_result_image(self, cv_image: np.ndarray, info: str = ""):
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        self._image = cv_image
        self._header.setText(self._title + (f"  |  {info}" if info else ""))
        self._update_display()

    def _cv2qt(self, cv_img):
        h, w = cv_img.shape[:2]
        ch = cv_img.shape[2] if len(cv_img.shape) > 2 else 1
        if ch == 4:
            fmt = QImage.Format_RGBA8888
            data = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        elif ch == 3:
            fmt = QImage.Format_RGB888
            data = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            fmt = QImage.Format_Grayscale8
            data = cv_img

        qimg = QImage(data.data, w, h, w * ch, fmt)
        return QPixmap.fromImage(qimg)

    def _update_display(self):
        if self._image is None:
            return

        pm = self._cv2qt(self._image)
        # fit-to-canvas base scale, then apply user zoom on top
        cw = max(self._canvas.width(), 1)
        ch = max(self._canvas.height(), 1)
        base = min(cw / pm.width(), ch / pm.height())
        self._scale = base * self._zoom
        new_w = max(1, int(pm.width() * self._scale))
        new_h = max(1, int(pm.height() * self._scale))
        pm = pm.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        canvas = QPixmap(pm.size())
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, pm)

        if self._annotate_mode:
            self._draw_template_hints(painter, pm.width(), pm.height())

        self._draw_keypoints(painter, pm.width(), pm.height())

        if self._face_boxes:
            self._draw_face_boxes(painter, pm.width(), pm.height())

        painter.end()

        self._canvas.setPixmap(canvas)
        self._canvas.setAlignment(Qt.AlignCenter)

    def _draw_template_hints(self, painter, w, h):
        if self._kp_idx >= len(KEYPOINT_TEMPLATE):
            return

        if self._image is None:
            return

        img_h, img_w = self._image.shape[:2]

        default_positions = self._get_default_positions(img_w, img_h)

        painter.setFont(QFont("Microsoft YaHei", 9))

        for i in range(self._kp_idx, len(KEYPOINT_TEMPLATE)):
            name, key = KEYPOINT_TEMPLATE[i]
            if key not in self._keypoints:
                if key in default_positions:
                    x, y = default_positions[key]
                    x = x * self._scale
                    y = y * self._scale
                    is_next = (i == self._kp_idx)
                    if is_next:
                        # The point the next click will place: highlighted.
                        painter.setPen(QPen(QColor(200, 90, 40, 220), 2, Qt.SolidLine))
                        painter.setBrush(QBrush(QColor(230, 130, 70, 120)))
                        painter.drawEllipse(QPointF(x, y), 9, 9)
                        painter.setPen(QPen(QColor(180, 70, 20), 1))
                        painter.drawText(int(x + 12), int(y + 4),
                                         f"下一个: {name}")
                    else:
                        painter.setPen(QPen(QColor(180, 180, 180, 120), 1, Qt.DashLine))
                        painter.setBrush(QBrush(QColor(200, 200, 200, 60)))
                        painter.drawEllipse(QPointF(x, y), 6, 6)
                        painter.setPen(QPen(QColor(150, 150, 150, 200), 1))
                        painter.drawText(int(x + 9), int(y + 3), name)

    def _get_default_positions(self, w, h):
        cx = w * 0.55
        return {
            "hairline_left": (w*0.40, h*0.12),
            "hairline_center": (cx, h*0.10),
            "hairline_right": (w*0.70, h*0.12),
            "brow_left_inner": (w*0.48, h*0.28),
            "brow_left_outer": (w*0.35, h*0.26),
            "eye_left_inner": (w*0.49, h*0.36),
            "eye_left_outer": (w*0.36, h*0.35),
            "nose_tip": (cx, h*0.48),
            "nose_left": (w*0.46, h*0.48),
            "mouth_left": (w*0.48, h*0.62),
            "mouth_center": (cx, h*0.63),
            "mouth_right": (w*0.62, h*0.62),
            "chin_tip": (cx, h*0.78),
            "chin_left": (w*0.45, h*0.75),
            "chin_right": (w*0.65, h*0.75),
            "ear_top": (w*0.30, h*0.35),
            "ear_bottom": (w*0.28, h*0.48),
        }

    def _draw_keypoints(self, painter, w, h):
        for name, (x, y) in self._keypoints.items():
            x_s = x * self._scale
            y_s = y * self._scale
            painter.setPen(QPen(QColor(138, 106, 74), 2))
            painter.setBrush(QBrush(QColor(138, 106, 74)))
            painter.drawEllipse(QPointF(x_s, y_s), 5, 5)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(int(x_s + 8), int(y_s - 8), name)

    def _draw_face_boxes(self, painter, w, h):
        painter.setPen(QPen(QColor(74, 106, 138), 2))
        for box in self._face_boxes:
            x1, y1, x2, y2 = [int(v * self._scale) for v in box[:4]]
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._annotate_mode and self._image is not None:
                pos = self._canvas.mapFromParent(event.pos())
                label_pos = self._canvas.mapFromGlobal(event.globalPos())
                pm = self._canvas.pixmap()
                if pm is None:
                    return

                pm_w = pm.width()
                pm_h = pm.height()
                label_w = self._canvas.width()
                label_h = self._canvas.height()

                offset_x = (label_w - pm_w) // 2 if pm_w < label_w else 0
                offset_y = (label_h - pm_h) // 2 if pm_h < label_h else 0

                px = label_pos.x() - offset_x
                py = label_pos.y() - offset_y

                if 0 <= px < pm_w and 0 <= py < pm_h:
                    img_x = px / self._scale
                    img_y = py / self._scale

                    if self._kp_idx < len(KEYPOINT_TEMPLATE):
                        name, key = KEYPOINT_TEMPLATE[self._kp_idx]
                        self._keypoints[key] = (img_x, img_y)
                        self._kp_idx += 1
                        self.point_clicked.emit(key, img_x, img_y)
                        self._update_display()
            else:
                self._pan_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self._pan_pos is not None:
            delta = event.globalPos() - self._pan_pos
            self._pan_pos = event.globalPos()
            self._offset += delta

    def mouseReleaseEvent(self, event):
        self._pan_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(4.0, self._zoom * 1.15)
        else:
            self._zoom = max(0.25, self._zoom / 1.15)
        self._update_display()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.jpg', '.png', '.bmp', '.webp', '.jpeg')):
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.load_image(img)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image is not None:
            self._update_display()

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._update_display()

    def clear(self):
        self._image = None
        self._display_pixmap = None
        self._keypoints = {}
        self._kp_idx = 0
        self._face_boxes = []
        self._canvas.setText("点击或拖入图像")
        self._canvas.setPixmap(QPixmap())
        self.image_loaded.emit(False)
