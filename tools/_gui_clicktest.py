# GUI 全控件无头点击测试: offscreen 实例化真实 MainWindow,
# 逐个点击全部按钮 (除会弹文件对话框的加载钮) + 拨全部滑块
# 任何断线/异常/阻塞都会现形
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys

sys.path.insert(0, ".")

from PySide6.QtWidgets import QApplication, QPushButton, QMessageBox

app = QApplication(sys.argv)

# 消息框改打印 (防 headless 阻塞)
QMessageBox.warning = staticmethod(
    lambda *a, **k: print("  [MSG-box]", a[1] if len(a) > 1 else ""))
QMessageBox.information = staticmethod(
    lambda *a, **k: print("  [MSG-box]", a[1] if len(a) > 1 else ""))
QMessageBox.critical = staticmethod(
    lambda *a, **k: print("  [MSG-box]", a[1] if len(a) > 1 else ""))

from src.gui.main_window import MainWindow

w = MainWindow()
panel = w._control_panel

print("== 按钮逐个点击 (跳过 ①加载: 会弹文件对话框) ==", flush=True)
fail = []
n_ok = 0
for btn in panel.findChildren(QPushButton):
    label = btn.text() or btn.objectName() or "?"
    if btn is panel._btn_load:
        print(f"  跳过: {label} (文件对话框)")
        continue
    try:
        btn.click()
        n_ok += 1
    except Exception as e:
        fail.append(f"{label}: {e}")
        print(f"  [FAIL] {label}: {e}")
print(f"按钮点击 {n_ok} 个成功, {len(fail)} 个失败", flush=True)

print("== 滑块拨动测试 ==", flush=True)
try:
    panel._yaw_slider.itemAt(1).widget().setValue(15)
    panel._pitch_slider.itemAt(1).widget().setValue(5)
    panel._roll_slider.itemAt(1).widget().setValue(-5)
    panel._mirror_slider.itemAt(1).widget().setValue(80)
    panel._tps_slider.itemAt(1).widget().setValue(60)
    print("  五滑块 setValue 全部生效 (无异常)", flush=True)
except Exception as e:
    print(f"  [FAIL] 滑块: {e}", flush=True)

print("== 参数读取测试 ==", flush=True)
try:
    pr = panel.get_params("real")
    print("  real 参数:", pr, flush=True)
    pa = panel.get_params("anime")
    print("  anime 参数:", pa, flush=True)
except Exception as e:
    print(f"  [FAIL] get_params: {e}", flush=True)

print("== 结果 ==", flush=True)
print("PASS" if not fail else f"FAIL: {fail}", flush=True)
