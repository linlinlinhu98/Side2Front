# GUI 全流程无头测试: 加载 cat.jpg -> 真实分类 -> 点 ④ -> 拦截实际命令
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys

sys.path.insert(0, ".")

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide6.QtCore import QProcess

app = QApplication(sys.argv)

# mock: 文件对话框返回 cat.jpg
CAT = os.path.abspath("4/cat.jpg")
QFileDialog.getOpenFileName = staticmethod(
    lambda *a, **k: (CAT, ""))

# mock: 消息框不阻塞
QMessageBox.warning = staticmethod(
    lambda *a, **k: print("  [MSG]", a[1] if len(a) > 1 else ""))
QMessageBox.information = staticmethod(
    lambda *a, **k: print("  [MSG]", a[1] if len(a) > 1 else ""))

# 拦截 QProcess.start: 记录命令, 不真启动
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

w = MainWindow()

print("== 步骤1: 加载 cat.jpg (真实 _on_load) ==", flush=True)
w._on_load()
print(f"  _original_path = {w._original_path}", flush=True)
print(f"  _curated_kps = {'有' if w._curated_kps else '无'}", flush=True)
print(f"  _reference = {'有' if w._reference else '无'}", flush=True)
print(f"  _ref_kind = {getattr(w, '_ref_kind', '未设置')!r}", flush=True)
print(f"  _kind_info (加载后) = {w._kind_info!r}", flush=True)

print("== 步骤2: 同步跑分类 (_detect_image_type) ==", flush=True)
kind, kps, info = w._detect_image_type()
print(f"  kind={kind}, info={info[:60]}", flush=True)
# 走真实回调 (设置 _mode/_kind_info)
w._on_type_done((kind, kps, info))
print(f"  _mode={w._mode!r}, _kind_info={w._kind_info[:60]!r}", flush=True)

print("== 步骤3: 点击 ④ (真实 _on_generate_sd) ==", flush=True)
w._on_generate_sd()
print(f"  实际命令 program = {started.get('program', '未启动!')}", flush=True)
args = started.get("args", [])
for a in args:
    print(f"    {a}", flush=True)
if not started:
    print("  [FAIL] QProcess.start 未被调用!", flush=True)

# 判定
script = started.get("args", [""])[0] if started.get("args") else ""
ok = "gen_frontal_refs" in script
print(f"== 判定: {'✓ 走动物通道' if ok else '✗ 未走动物通道!'} ==", flush=True)
