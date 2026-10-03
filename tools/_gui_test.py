# GUI 无头测试合集 (合并自 _gui_clicktest.py + _gui_flow_test.py).
# usage: python tools/_gui_test.py [click|flow]
#   click (default): 实例化真实 MainWindow, 逐个点击全部按钮
#                    (除会弹文件对话框的加载钮) + 拨全部滑块,
#                    任何断线/异常/阻塞都会现形
#   flow:            加载 cat.jpg -> 真实分类 -> 点 ④ -> 拦截
#                    实际命令, 判定是否走动物通道
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys

sys.path.insert(0, ".")

from PySide6.QtWidgets import (QApplication, QPushButton,
                               QMessageBox, QFileDialog)
from PySide6.QtCore import QProcess

app = QApplication(sys.argv)

# 消息框改打印 (防 headless 阻塞)
QMessageBox.warning = staticmethod(
    lambda *a, **k: print("  [MSG-box]", a[1] if len(a) > 1 else ""))
QMessageBox.information = staticmethod(
    lambda *a, **k: print("  [MSG-box]", a[1] if len(a) > 1 else ""))
QMessageBox.critical = staticmethod(
    lambda *a, **k: print("  [MSG-box]", a[1] if len(a) > 1 else ""))

# 拦截 QProcess.start: 记录命令, 不真启动 (click 模式下也生效,
# 防止点到 SD 按钮真的拉起分钟级生图进程)
started = {}
_orig_start = QProcess.start


def fake_start(self, program, arguments=None, mode=None):
    started["program"] = program
    started["args"] = [str(a) for a in (arguments or [])]
    print(f"  [QProcess 启动] {program}")
    for a in started["args"]:
        print(f"    arg: {a}")
    return


QProcess.start = fake_start

from src.gui.main_window import MainWindow

MODE = "flow" if len(sys.argv) > 1 and sys.argv[1] == "flow" \
    else "click"

w = MainWindow()

if MODE == "click":
    panel = w._control_panel

    print("== 按钮逐个点击 (跳过 ①加载: 会弹文件对话框) ==",
          flush=True)
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

else:  # flow
    # mock: 文件对话框返回 cat.jpg
    CAT = os.path.abspath("4/cat.jpg")
    QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (CAT, ""))

    print("== 步骤1: 加载 cat.jpg (真实 _on_load) ==", flush=True)
    w._on_load()
    print(f"  _original_path = {w._original_path}", flush=True)
    print(f"  _curated_kps = {'有' if w._curated_kps else '无'}",
          flush=True)
    print(f"  _reference = {'有' if w._reference else '无'}",
          flush=True)
    print(f"  _ref_kind = {getattr(w, '_ref_kind', '未设置')!r}",
          flush=True)
    print(f"  _kind_info (加载后) = {w._kind_info!r}", flush=True)

    print("== 步骤2: 同步跑分类 (_detect_image_type) ==", flush=True)
    kind, kps, info = w._detect_image_type()
    print(f"  kind={kind}, info={info[:60]}", flush=True)
    # 走真实回调 (设置 _mode/_kind_info)
    w._on_type_done((kind, kps, info))
    print(f"  _mode={w._mode!r}, "
          f"_kind_info={w._kind_info[:60]!r}", flush=True)

    print("== 步骤3: 点击 ④ (真实 _on_generate_sd) ==", flush=True)
    w._on_generate_sd()
    print(f"  实际命令 program = "
          f"{started.get('program', '未启动!')}", flush=True)
    args = started.get("args", [])
    for a in args:
        print(f"    {a}", flush=True)
    if not started:
        print("  [FAIL] QProcess.start 未被调用!", flush=True)

    # 判定
    script = started.get("args", [""])[0] \
        if started.get("args") else ""
    ok = "gen_frontal_refs" in script
    print(f"== 判定: "
          f"{'[OK] 走动物通道' if ok else '[NG] 未走动物通道!'} ==",
          flush=True)
