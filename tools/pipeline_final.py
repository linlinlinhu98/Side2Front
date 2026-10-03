"""PIPELINE FINAL - one-command deterministic replay of the
accepted chain that produces the FINAL frontal image
(genr156_s42.png == assets/final/face_frontal.png) from the
original side profile 4/face.png.

The final image is NOT one-shot generation. It is a
13-stage deterministic pipeline; every stage uses a FIXED
seed and reads exactly the previous stage's accepted output,
so a replay is bit-identical (each stage ran as its own
process originally - same conditions here).

Stage list (module, accepted output, what it does):
  1. gen_r119   genr119_s7    base generation: img2img FROM
                             the 8aa reference (structure)
                             + IP-Adapter identity swap to
                             the original boy
  2. gen_r145   genr145_s42   right vertical strap replaces
                             the diagonal one
  3. gen_r146   genr146_s42   left strap square buckle
  4. gen_r147   genr147_s42   zipper + ring pull
  5. gen_r148   genr148_s42   right strap hatch texture
  6. gen_r149   genr149_s7    erase the phantom drawstring
  7. gen_r150   genr150_s42   right strap shifted to x~408
  8. gen_r151   genr151_s42   straight zipper, ladder teeth
                             full length
  9. gen_r152b  genr152b_s42  thin-line cleanup right collar
 10. gen_r153   genr153_s42   right strap top widened + ink
                             blob removed
 11. gen_r154   genr154_s42   right strap uniform 48px
 12. gen_r155   genr155_s7    left strap uniform + buckle
                             hardware removed
 13. gen_r156   genr156_s42   MIRROR TRANSPLANT (pixel op):
                             right strap flipped onto the
                             left -> exact symmetry

Usage (from repo root):
  HF_HOME=D:\\huggingface_cache HF_HUB_OFFLINE=1 \\
      /d/Python/python.exe tools/pipeline_final.py [--force]

Resumable: a stage is skipped if its accepted output already
exists. --force re-runs everything. Each stage runs as a
SEPARATE python process (the exact conditions the chain was
built in).

FAST replay (default): each stage runs ONLY its accepted
seed (injected via S2F_SEEDS) - the non-accepted seeds were
never used by the chain, so the replayed outputs are
bit-identical while total time drops from ~3h to ~1.5h CPU.
To also regenerate the rejected seeds (as in the original
tuning sessions), run a stage directly without S2F_SEEDS.
"""
import os
import subprocess
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402

STAGES = [
    ("gen_r119", "genr119_s7.png",
     "base: 8aa-start img2img + IP identity swap"),
    ("gen_r145", "genr145_s42.png", "right vertical strap"),
    ("gen_r146", "genr146_s42.png", "left strap buckle"),
    ("gen_r147", "genr147_s42.png", "zipper + ring pull"),
    ("gen_r148", "genr148_s42.png", "strap hatch texture"),
    ("gen_r149", "genr149_s7.png", "erase drawstring"),
    ("gen_r150", "genr150_s42.png", "right strap to x~408"),
    ("gen_r151", "genr151_s42.png", "zipper ladder teeth"),
    ("gen_r152b", "genr152b_s42.png", "collar thin-line clean"),
    ("gen_r153", "genr153_s42.png", "strap top + blob fix"),
    ("gen_r154", "genr154_s42.png", "right strap uniform 48"),
    ("gen_r155", "genr155_s7.png", "left strap uniform"),
    ("gen_r156", "genr156_s42.png", "mirror transplant"),
]


def _env():
    e = os.environ.copy()
    e.setdefault("HF_HOME", r"D:\huggingface_cache")
    e.setdefault("HF_HUB_OFFLINE", "1")
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return e


def main():
    force = "--force" in sys.argv
    total = time.time()
    for i, (mod, out_name, desc) in enumerate(STAGES, 1):
        out_path = os.path.join(OUT, out_name)
        if os.path.exists(out_path) and not force:
            print(f"[pipeline {i:2d}/13] SKIP {mod} "
                  f"({out_name} exists)", flush=True)
            continue
        print(f"[pipeline {i:2d}/13] RUN  {mod} - {desc}",
              flush=True)
        t0 = time.time()
        env = _env()
        # only the accepted seed matters for the replay; the
        # rejected one was never consumed by any later stage
        env["S2F_SEEDS"] = \
            os.path.splitext(out_name)[0].rsplit("_s", 1)[-1]
        r = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "stages", f"{mod}.py")],
            env=env, cwd=ROOT)
        if r.returncode != 0 or not os.path.exists(out_path):
            print(f"[pipeline] FAILED at {mod} "
                  f"(rc={r.returncode})", flush=True)
            sys.exit(1)
        print(f"[pipeline {i:2d}/13] done {mod} in "
              f"{time.time() - t0:.0f}s", flush=True)

    final = os.path.join(OUT, "genr156_s42.png")
    archive = os.path.join(ROOT, "assets", "final",
                           "face_frontal.png")
    print(f"[pipeline] total {time.time() - total:.0f}s",
          flush=True)
    if os.path.exists(archive):
        a = cv2.imread(archive, 0)
        b = cv2.imread(final, 0)
        if a is not None and b is not None \
                and a.shape == b.shape:
            diff = (a != b).mean()
            print(f"[pipeline] replay vs archived final: "
                  f"pixel diff = {diff:.6f} "
                  f"({'BIT-IDENTICAL' if diff == 0 else 'DRIFT'})",
                  flush=True)
    else:
        cv2.imwrite(archive, cv2.imread(final, 0))
        print(f"[pipeline] archived -> {archive}", flush=True)


if __name__ == "__main__":
    main()
