from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QSlider, QLabel,
    QButtonGroup, QFrame, QFileDialog, QGroupBox, QHBoxLayout
)
from PySide6.QtCore import Qt, Signal


STYLE_SHEET = """
QWidget { font-size: 12px; color: #444; }
QGroupBox {
    border: 1px solid #ddd; border-radius: 4px;
    margin-top: 10px; padding-top: 14px;
    font-size: 11px; color: #888; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
}
QPushButton {
    padding: 6px 10px; border: 1px solid #ccc;
    border-radius: 3px; background: #f5f5f5; color: #444;
    text-align: left; font-size: 12px;
}
QPushButton:hover { background: #e8e8e8; }
QPushButton:pressed { background: #ddd; }
QPushButton:disabled { color: #aaa; border-color: #eee; background: #f8f8f8; }
QPushButton#primary {
    background: #333; color: #fff; border-color: #333;
}
QPushButton#primary:hover { background: #555; }
QPushButton#primary:disabled { background: #aaa; border-color: #aaa; }
QPushButton#mode_btn {
    text-align: center; font-weight: 600; padding: 7px 4px;
}
QPushButton#mode_btn:checked {
    background: #333; color: #fff; border-color: #333;
}
QSlider::groove:horizontal {
    height: 4px; background: #ddd; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0;
    background: #555; border-radius: 7px;
}
QLabel { background: transparent; }
"""


class ControlPanel(QWidget):
    load_clicked = Signal()
    convert_clicked = Signal()
    export_clicked = Signal()
    generate_sd_clicked = Signal()
    register_final_clicked = Signal()
    change_batch_clicked = Signal()
    fix_clicked = Signal(str)
    prev_version_clicked = Signal()
    next_version_clicked = Signal()
    params_changed = Signal()
    annotate_clicked = Signal(bool)
    auto_annotate_clicked = Signal()
    clear_points_clicked = Signal()
    compare_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._compare_mode = False
        self._init_ui()
        self.setStyleSheet(STYLE_SHEET)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        file_layout.setContentsMargins(4, 4, 4, 4)
        self._btn_load = QPushButton("① 加载图像")
        self._btn_load.setObjectName("primary")
        self._btn_convert = QPushButton("② 开始转换")
        self._btn_convert.setObjectName("primary")
        self._btn_convert.setEnabled(False)
        self._btn_export = QPushButton("③ 导出结果")
        self._btn_export.setEnabled(False)
        self._btn_gen_sd = QPushButton("④ SD 生成正脸")
        self._btn_gen_sd.setEnabled(False)
        self._btn_register = QPushButton("⑤ 注册为定稿")
        self._btn_register.setEnabled(False)
        self._btn_batch = QPushButton("⑥ 换一批（新种子重生成）")
        self._btn_batch.setEnabled(False)
        file_layout.addWidget(self._btn_load)
        file_layout.addWidget(self._btn_convert)
        file_layout.addWidget(self._btn_export)
        file_layout.addWidget(self._btn_gen_sd)
        file_layout.addWidget(self._btn_batch)
        file_layout.addWidget(self._btn_register)
        layout.addWidget(file_group)

        self._btn_load.clicked.connect(self.load_clicked.emit)
        self._btn_convert.clicked.connect(self.convert_clicked.emit)
        self._btn_export.clicked.connect(self.export_clicked.emit)
        self._btn_gen_sd.clicked.connect(self.generate_sd_clicked.emit)
        self._btn_batch.clicked.connect(self.change_batch_clicked.emit)
        self._btn_register.clicked.connect(self.register_final_clicked.emit)

        # 部位重修行: 四个平铺按钮, 点哪个修哪个 (无嵌套)
        fix_row = QHBoxLayout()
        fix_row.addWidget(QLabel("部位重修:"))
        self._btn_fix_nose = QPushButton("鼻")
        self._btn_fix_mouth = QPushButton("嘴")
        self._btn_fix_teeth = QPushButton("牙")
        self._btn_fix_jaw = QPushButton("下颌")
        for b in (self._btn_fix_nose, self._btn_fix_mouth,
                  self._btn_fix_teeth, self._btn_fix_jaw):
            b.setEnabled(False)
            fix_row.addWidget(b)
        layout.addLayout(fix_row)
        self._btn_fix_nose.clicked.connect(
            lambda: self.fix_clicked.emit("nose"))
        self._btn_fix_mouth.clicked.connect(
            lambda: self.fix_clicked.emit("mouth"))
        self._btn_fix_teeth.clicked.connect(
            lambda: self.fix_clicked.emit("teeth"))
        self._btn_fix_jaw.clicked.connect(lambda: self.fix_clicked.emit("jaw"))

        # 版本历史行: 前后翻页
        ver_row = QHBoxLayout()
        ver_row.addWidget(QLabel("版本:"))
        self._btn_prev_ver = QPushButton("◀ 上一版")
        self._btn_prev_ver.setEnabled(False)
        self._ver_label = QLabel("无")
        self._btn_next_ver = QPushButton("下一版 ▶")
        self._btn_next_ver.setEnabled(False)
        ver_row.addWidget(self._btn_prev_ver)
        ver_row.addWidget(self._ver_label)
        ver_row.addWidget(self._btn_next_ver)
        layout.addLayout(ver_row)
        self._btn_prev_ver.clicked.connect(self.prev_version_clicked.emit)
        self._btn_next_ver.clicked.connect(self.next_version_clicked.emit)

        self._real_group = QGroupBox("旋转参数 (3DMM)")
        real_layout = QVBoxLayout(self._real_group)
        real_layout.setContentsMargins(4, 4, 4, 4)

        self._yaw_slider, self._yaw_label = self._make_slider("Yaw (左右转)", -90, 90, 0, "°")
        real_layout.addLayout(self._yaw_slider)
        self._pitch_slider, self._pitch_label = self._make_slider("Pitch (俯仰)", -30, 30, 0, "°")
        real_layout.addLayout(self._pitch_slider)
        self._roll_slider, self._roll_label = self._make_slider("Roll (歪头)", -30, 30, 0, "°")
        real_layout.addLayout(self._roll_slider)

        self._btn_reset_angles = QPushButton("重置角度")
        real_layout.addWidget(self._btn_reset_angles)

        self._btn_texture = QPushButton("纹理补全: 开启")
        self._btn_texture.setCheckable(True)
        self._btn_texture.setChecked(True)
        real_layout.addWidget(self._btn_texture)

        self._btn_smooth = QPushButton("边界平滑: 开启")
        self._btn_smooth.setCheckable(True)
        self._btn_smooth.setChecked(True)
        real_layout.addWidget(self._btn_smooth)

        layout.addWidget(self._real_group)

        self._btn_reset_angles.clicked.connect(self._reset_angles)

        self._anime_group = QGroupBox("关键点标注")
        anime_layout = QVBoxLayout(self._anime_group)
        anime_layout.setContentsMargins(4, 4, 4, 4)

        self._btn_annotate = QPushButton("手动标注关键点")
        self._btn_annotate.setCheckable(True)
        anime_layout.addWidget(self._btn_annotate)

        self._btn_auto_annotate = QPushButton("自动检测关键点")
        anime_layout.addWidget(self._btn_auto_annotate)

        self._btn_clear_points = QPushButton("清除所有标注")
        anime_layout.addWidget(self._btn_clear_points)

        anime_layout.addSpacing(4)

        self._mirror_slider, self._mirror_label = self._make_slider("镜像强度", 0, 100, 100, "%")
        anime_layout.addLayout(self._mirror_slider)

        self._tps_slider, self._tps_label = self._make_slider("变形强度", 0, 100, 100, "%")
        anime_layout.addLayout(self._tps_slider)

        self._btn_reset_anime = QPushButton("重置参数")
        anime_layout.addWidget(self._btn_reset_anime)

        self._btn_poisson = QPushButton("泊松融合: 开启")
        self._btn_poisson.setCheckable(True)
        self._btn_poisson.setChecked(True)
        anime_layout.addWidget(self._btn_poisson)

        self._btn_color_match = QPushButton("色调匹配: 开启")
        self._btn_color_match.setCheckable(True)
        self._btn_color_match.setChecked(True)
        anime_layout.addWidget(self._btn_color_match)

        layout.addWidget(self._anime_group)
        self._anime_group.hide()

        self._btn_annotate.clicked.connect(
            lambda checked: self.annotate_clicked.emit(checked))
        self._btn_auto_annotate.clicked.connect(self.auto_annotate_clicked.emit)
        self._btn_clear_points.clicked.connect(self.clear_points_clicked.emit)
        self._btn_reset_anime.clicked.connect(self._reset_anime)

        display_group = QGroupBox("显示")
        display_layout = QVBoxLayout(display_group)
        display_layout.setContentsMargins(4, 4, 4, 4)
        self._btn_compare = QPushButton("对比模式: 关闭")
        self._btn_compare.setCheckable(True)
        display_layout.addWidget(self._btn_compare)
        layout.addWidget(display_group)

        self._btn_compare.clicked.connect(self._toggle_compare)

        for btn in [self._btn_texture, self._btn_smooth, self._btn_poisson, self._btn_color_match]:
            btn.clicked.connect(self._on_toggle_btn)

        self._yaw_slider.itemAt(1).widget().valueChanged.connect(self._on_yaw)
        self._pitch_slider.itemAt(1).widget().valueChanged.connect(self._on_pitch)
        self._roll_slider.itemAt(1).widget().valueChanged.connect(self._on_roll)
        self._mirror_slider.itemAt(1).widget().valueChanged.connect(self._on_mirror)
        self._tps_slider.itemAt(1).widget().valueChanged.connect(self._on_tps)

        layout.addStretch()

    def _make_slider(self, name, mn, mx, val, suffix):
        container = QVBoxLayout()
        container.setSpacing(2)
        label_row = QHBoxLayout()
        left_label = QLabel(name)
        left_label.setStyleSheet("font-size:11px; color:#666;")
        val_label = QLabel(f"{val}{suffix}")
        val_label.setStyleSheet("font-size:11px; color:#333; font-weight:600;")
        val_label.setAlignment(Qt.AlignRight)
        label_row.addWidget(left_label)
        label_row.addStretch()
        label_row.addWidget(val_label)
        container.addLayout(label_row)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(val)
        slider._suffix = suffix
        container.addWidget(slider)

        return container, val_label

    def _on_yaw(self, v):
        self._yaw_label.setText(f"{v}°")
        self.params_changed.emit()

    def _on_pitch(self, v):
        self._pitch_label.setText(f"{v}°")
        self.params_changed.emit()

    def _on_roll(self, v):
        self._roll_label.setText(f"{v}°")
        self.params_changed.emit()

    def _on_mirror(self, v):
        self._mirror_label.setText(f"{v}%")
        self.params_changed.emit()

    def _on_tps(self, v):
        self._tps_label.setText(f"{v}%")
        self.params_changed.emit()

    def _on_toggle_btn(self):
        btn = self.sender()
        text = btn.text().split(":")[0]
        state = "开启" if btn.isChecked() else "关闭"
        btn.setText(f"{text}: {state}")
        self.params_changed.emit()

    def _toggle_compare(self):
        self._compare_mode = not self._compare_mode
        state = "开启" if self._compare_mode else "关闭"
        self._btn_compare.setText(f"对比模式: {state}")
        self.compare_toggled.emit(self._compare_mode)

    def _reset_angles(self):
        self._yaw_slider.itemAt(1).widget().setValue(0)
        self._pitch_slider.itemAt(1).widget().setValue(0)
        self._roll_slider.itemAt(1).widget().setValue(0)

    def _reset_anime(self):
        self._mirror_slider.itemAt(1).widget().setValue(100)
        self._tps_slider.itemAt(1).widget().setValue(100)

    def switch_mode(self, mode: str):
        """Show the parameter group matching the auto-detected type."""
        if mode == "real":
            self._real_group.show()
            self._anime_group.hide()
        else:
            self._real_group.hide()
            self._anime_group.show()

    def set_convert_enabled(self, enabled: bool):
        self._btn_convert.setEnabled(enabled)

    def set_export_enabled(self, enabled: bool):
        self._btn_export.setEnabled(enabled)

    def set_batch_enabled(self, enabled: bool):
        self._btn_batch.setEnabled(enabled)

    def set_fix_enabled(self, enabled: bool):
        for b in (self._btn_fix_nose, self._btn_fix_mouth,
                  self._btn_fix_teeth, self._btn_fix_jaw):
            b.setEnabled(enabled)

    def set_version_nav(self, prev_on: bool, next_on: bool, label: str):
        self._btn_prev_ver.setEnabled(prev_on)
        self._btn_next_ver.setEnabled(next_on)
        self._ver_label.setText(label)

    def set_gen_sd_enabled(self, enabled: bool):
        self._btn_gen_sd.setEnabled(enabled)

    def set_register_enabled(self, enabled: bool):
        self._btn_register.setEnabled(enabled)

    def get_params(self, mode: str) -> dict:
        if mode == "real":
            return {
                "yaw": self._yaw_slider.itemAt(1).widget().value(),
                "pitch": self._pitch_slider.itemAt(1).widget().value(),
                "roll": self._roll_slider.itemAt(1).widget().value(),
                "texture_completion": self._btn_texture.isChecked(),
                "edge_smooth": self._btn_smooth.isChecked(),
            }
        else:
            return {
                "mirror_strength": self._mirror_slider.itemAt(1).widget().value() / 100.0,
                "tps_flexibility": self._tps_slider.itemAt(1).widget().value() / 100.0,
                "poisson_blend": self._btn_poisson.isChecked(),
                "color_match": self._btn_color_match.isChecked(),
            }

    def set_annotate_btn_checked(self, checked: bool):
        self._btn_annotate.setChecked(checked)

    def set_real_pose(self, yaw: float, pitch: float, roll: float):
        """Update the Yaw/Pitch/Roll sliders to detected pose values."""
        self._yaw_slider.itemAt(1).widget().setValue(int(round(yaw)))
        self._pitch_slider.itemAt(1).widget().setValue(int(round(pitch)))
        self._roll_slider.itemAt(1).widget().setValue(int(round(roll)))
        self._yaw_label.setText(f"{int(round(yaw))}°")
        self._pitch_label.setText(f"{int(round(pitch))}°")
        self._roll_label.setText(f"{int(round(roll))}°")
