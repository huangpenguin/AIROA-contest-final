# T2 H10K 明早 persist / fold / Stop 执行稿

> 落盘：2026-09-13 夜（Asia/Tokyo），供 2026-09-14 早上现场执行
> 读者：官方 Jupyter 现场 agent
> 配套合同：`docs/AXIS_PI05_T2_H10K_POST_TRAIN_AGENT_HANDOFF_20260913.md`
> 启动问题：`docs/AXIS_PI05_OFFICIAL_BLACKWELL_T2_H10K_LAUNCH_ISSUES_20260913.md`
> 机器 JSON：`data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_tomorrow_persist_fold_20260914/summary.json`

今天的经验：不要 Restart and run all；不要把两行 shell 黏在一起；Python cell 比 bash 粘贴稳；`find -printf` 可在 Jupyter **终端**用；训练还在写盘时不要 copy 正在写的 step；没 data-push 就 Stop = checkpoint 消失。

本文件给出仓库里能钉死的命令。**fold 是否缺 torch、Docker 镜像真实名字，必须现场 `import` / `docker images` 核对后再跑。** 不要删文件，不要在 persist 未核 SHA 前 Stop My Server。Agent 不自动上传/提交。

---

## 0. 冻结路径

```text
RUN_ID     t2_p30_r6_h10k_r1
CODE       /opt/dlami/nvme/parc20260911/code/PARC2026_final
NVME_RUN   /opt/dlami/nvme/parc20260911/runs/t2_p30_r6_h10k_r1
LOG        /opt/dlami/nvme/parc20260911/runs/t2_p30_r6_h10k_r1.train.log
PYTHON     /opt/dlami/nvme/parc20260911/work/trainvenv_python.sh
PERSIST    $HOME/data/parc20260911/t2_p30_r6_h10k_r1
NORM_SRC   $CODE/data/manifests_track12/T2-P30/norm_stats.json
NORM_SHA   39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41
KEEP       1 10 100 1000 3000 10000
```

`~/data` 100 GiB，需 `parc-home-sync data-push` 才过得了 Stop。`$HOME` 40 GiB 不要放大 safetensors。完整 Orbax（params+train_state）可能十几到几十 GiB，**只推最高完整 step**，不要 1000+3000+10000 全推。

---

## 1. 先只读（Jupyter 新 cell，Shift+Enter）

不要 Restart and run all。`unset LD_LIBRARY_PATH`。

```python
import os, json, hashlib, subprocess
from pathlib import Path
os.environ.pop("LD_LIBRARY_PATH", None)

RUN_ID = "t2_p30_r6_h10k_r1"
nvme_run = Path("/opt/dlami/nvme/parc20260911/runs") / RUN_ID
log = Path(str(nvme_run) + ".train.log")
persist = Path.home() / "data/parc20260911" / RUN_ID
code = Path("/opt/dlami/nvme/parc20260911/code/PARC2026_final")
keep = (1, 10, 100, 1000, 3000, 10000)

print("train.py")
print(subprocess.run(["pgrep", "-af", "scripts/train.py"], capture_output=True, text=True).stdout.strip() or "(none)")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv"], text=True))
print("log_bytes", log.stat().st_size if log.exists() else None)
if log.exists():
    lines = log.read_text(errors="replace").splitlines()
    print("\n".join(lines[-12:]))

def complete(step: Path) -> bool:
    return (step / "params").is_dir() and (step / "train_state").is_dir() and (step / "_CHECKPOINT_METADATA").is_file()

print("run_dir_children")
if nvme_run.is_dir():
    for child in sorted(nvme_run.iterdir(), key=lambda p: p.name):
        if child.is_dir() and child.name.isdigit():
            print(" ", child.name, "complete", complete(child), "bytes", sum(f.stat().st_size for f in child.rglob("*") if f.is_file()))
        elif child.is_dir():
            print(" ", child.name, "(not a keep step)")

complete_keep = [s for s in keep if complete(nvme_run / str(s))]
print("complete_keep", complete_keep)
print("last_complete_step", complete_keep[-1] if complete_keep else None)

ident = nvme_run / "run_identity.json"
print("identity_exists", ident.is_file())
if ident.is_file():
    print(ident.read_text()[:800])

print("persist", persist)
print("df")
print(subprocess.check_output(["df", "-h", str(Path.home()), str(Path.home() / "data"), "/opt/dlami/nvme"], text=True))
print("parc-home-sync", subprocess.run(["parc-home-sync", "--help"], capture_output=True, text=True).stdout[:1500] or subprocess.run(["parc-home-sync"], capture_output=True, text=True).stderr[:800])
```

判据：

- 训练还在跑：只观察；可 copy **已经 complete 的 KEEP step**（例如 1000/3000），不要 copy 正在写的最高目录。
- 训练已停、GPU 接近 0 MiB：选 `last_complete_step`，优先 10000，否则 3000，否则 1000。
- 只有 `params` 没有 `train_state`：可以 fold，**不能 resume**。标在报告里。
- `~/data` 剩余必须大于 `du -sh SRC` 再加 folded ~15 GiB 余量。不够就只推 params + metadata + log，train_state 次之（resume 会牺牲）。默认仍尽量 params+train_state 一起推。

---

## 2. 复制到 persist 再 push（确认 STEP 之后）

把 `STEP` 换成上一格打印的 `last_complete_step`。Jupyter **终端**（不是 notebook Restart）：

```bash
unset LD_LIBRARY_PATH
export RUN_ID=t2_p30_r6_h10k_r1
export STEP=10000   # 改成现场 last_complete_step
export NVME_RUN=/opt/dlami/nvme/parc20260911/runs/$RUN_ID
export SRC="$NVME_RUN/$STEP"
export PERSIST_DIR="$HOME/data/parc20260911/$RUN_ID"
export DST="$PERSIST_DIR/checkpoints/$STEP"
export STAGE="$PERSIST_DIR/.staging/$STEP"
export CODE=/opt/dlami/nvme/parc20260911/code/PARC2026_final

test -d "$SRC/params"
test -d "$SRC/train_state"
test -f "$SRC/_CHECKPOINT_METADATA"
du -sh "$SRC"
df -h "$HOME/data"

mkdir -p "$PERSIST_DIR/checkpoints" "$PERSIST_DIR/identity" "$PERSIST_DIR/folded"
test ! -e "$DST"
rm -rf "$STAGE"
mkdir -p "$STAGE"
rsync -a --info=progress2 "$SRC/" "$STAGE/"
test -d "$STAGE/params"
test -d "$STAGE/train_state"
test -f "$STAGE/_CHECKPOINT_METADATA"
mv "$STAGE" "$DST"

cp -p "$NVME_RUN.train.log" "$PERSIST_DIR/" || true
test -f "$NVME_RUN/run_identity.json" && cp -p "$NVME_RUN/run_identity.json" "$PERSIST_DIR/identity/"
cp -p "$CODE/data/manifests_track12/T2-P30/norm_stats.json" "$PERSIST_DIR/norm_stats.json"
python3 - <<'PY'
import hashlib, json
from pathlib import Path
p = Path.home()/"data/parc20260911/t2_p30_r6_h10k_r1/norm_stats.json"
h = hashlib.sha256(p.read_bytes()).hexdigest()
exp = "39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41"
print("norm_sha", h, "match", h==exp)
if h != exp:
    raise SystemExit("norm sha mismatch")
PY

parc-home-sync data-push
```

关于 `parc-home-sync data-push`：仓库与运营资料写的是 **只同步 `~/data`**（100 GiB 持久盘），不是 `$HOME` 40 GiB，也不是 NVMe。现场仍先跑 `parc-home-sync --help` / `data-ls` 核对。push 完用 `data-ls` 看到 `t2_p30_r6_h10k_r1/checkpoints/$STEP/_CHECKPOINT_METADATA` 才算过。

若 `DST` 已存在：先 `mv "$DST" "$DST.bak_$(date +%H%M%S)"`，不要 `rm -rf`。

---

## 3. fold（必须官方 JAX 0.11.1；先探测再跑）

仓库入口是 `tools.axis_pi05.convert_for_lerobot`：从 `$SRC/params` restore JAX 树、fold LoRA、写成 `model.safetensors`。必须用 **trainvenv wrapper**，不要 `Path.resolve()` 那个 uv symlink。

**探测（新 cell，不改 venv）：**

```python
import os, subprocess
from pathlib import Path
os.environ.pop("LD_LIBRARY_PATH", None)
py = "/opt/dlami/nvme/parc20260911/work/trainvenv_python.sh"
code = "/opt/dlami/nvme/parc20260911/code/PARC2026_final"
r = subprocess.run(
    [py, "-c", "import jax, flax, orbax; print('jax', jax.__version__);\n"
     "mods=[]\n"
     "for m in ('torch','safetensors','numpy'):\n"
     "    try:\n"
     "        __import__(m); mods.append(m+':ok')\n"
     "    except Exception as e:\n"
     "        mods.append(m+':'+type(e).__name__)\n"
     "print(' '.join(mods))"],
    cwd=code, text=True, capture_output=True, env={k:v for k,v in os.environ.items() if k!="LD_LIBRARY_PATH"},
)
print("rc", r.returncode)
print(r.stdout)
print(r.stderr[-800:])
```

若 `torch:ok` 且 `safetensors:ok`，在 **GPU 已空、train.py 已退出** 后跑：

```bash
unset LD_LIBRARY_PATH
export STEP=10000   # 与 persist 的 STEP 相同
export SRC=/opt/dlami/nvme/parc20260911/runs/t2_p30_r6_h10k_r1/$STEP
export OUT=/opt/dlami/nvme/parc20260911/work/t2_p30_r6_h10k_r1_folded_$STEP
export PERSIST=$HOME/data/parc20260911/t2_p30_r6_h10k_r1
export CODE=/opt/dlami/nvme/parc20260911/code/PARC2026_final

test ! -e "$OUT"
/opt/dlami/nvme/parc20260911/work/trainvenv_python.sh -m tools.axis_pi05.convert_for_lerobot \
  --checkpoint "$SRC" \
  --output "$OUT"

test -f "$OUT/model.safetensors"
python3 - <<PY
import hashlib, json, os
from pathlib import Path
step = os.environ["STEP"]
src = Path(os.environ["OUT"])
dst_dir = Path.home()/f"data/parc20260911/t2_p30_r6_h10k_r1/folded/{step}"
dst_dir.mkdir(parents=True, exist_ok=True)
model = src/"model.safetensors"
data = model.read_bytes()
sha = hashlib.sha256(data).hexdigest()
print("bytes", len(data), "sha256", sha)
# copy via replace
import shutil
shutil.copy2(model, dst_dir/"model.safetensors")
if (src/"config.json").is_file():
    shutil.copy2(src/"config.json", dst_dir/"config.json")
(dst_dir/"sha256.json").write_text(json.dumps({"path":"model.safetensors","bytes":len(data),"sha256":sha}, indent=2)+"\n")
PY
cp -p "$PERSIST/norm_stats.json" "$PERSIST/folded/$STEP/" 2>/dev/null || true
parc-home-sync data-push
```

若探测里 `torch:ModuleNotFoundError`：**不要往 trainvenv 装 PyTorch**（会冲 JAX 0.11.1）。先把 Orbax checkpoint + norm + identity push 完，fold 标 `blocked_missing_torch`。下一 session 再用官方 pre-check Docker（PyTorch 2.11）做转换，或单独建 fold venv，须新审核。

`convert_for_lerobot` 会读 `$SRC/params`。官方 checkpoint 是 JAX 0.11 / Orbax 0.11.25，**不能拿到本机 jax 0.5.3 上 restore**。

---

## 4. 什么时候可以 Stop My Server

| 已 push | 能否 Stop | 后果 |
|---|---|---|
| 只有 NVMe、没 push | **否** | 清 NVMe，重演昨晚 pid 210633 |
| checkpoint（params+train_state+metadata）+ log + identity + norm SHA | 可以 Stop，但下次还要开官方机 fold | 权重不丢，fold 要 JAX 0.11 |
| 上面 + `model.safetensors` + SHA | **推荐 Stop** | 本机可 C20 / FAST / ZIP |
| 训练进程还在 / GPU 非空 | **否** | 先等退出 |

Stop 前再跑：

```text
pgrep -af scripts/train.py     → 空
nvidia-smi memory.used         → ~0 MiB
parc-home-sync data-ls | grep t2_p30_r6_h10k_r1
```

然后网页 **File → Hub Control Panel → Stop My Server**。关浏览器标签 **不等于** 停服务器，idle 1h / 满 12h 仍会停。

---

## 5. Docker pre-check（次优先，不能替代 C20）

仓库 README 占位名是 `parc2026`，AMI 上真实名字现场 `docker images` 为准。不要 `docker build`，不要 custom image。

资料口径（须现场核对 image 名）：

```bash
docker images
# 交互壳：把 ~/data 挂到 /data
docker run -it --rm --gpus all -v ~/data:/data parc2026
# 容器内（ZIP 已在 ~/data/... 时）：
# python evaluate.py /data/<submission.zip> --n-episodes 2
```

时间紧：官方只做 checkpoint/persist/fold；C20 和 FAST ZIP 回本机 GPU3。Docker 只做 ZIP 已存在后的启动/2-episode smoke。不要在 trainvenv 里装 LIBERO。

正式 scorer 是单张 L4 24GB，不是 Blackwell。

---

## 6. persist 目录约定

```text
~/data/parc20260911/t2_p30_r6_h10k_r1/
  checkpoints/<STEP>/          Orbax params + train_state + _CHECKPOINT_METADATA
  folded/<STEP>/model.safetensors
  folded/<STEP>/config.json
  folded/<STEP>/sha256.json
  identity/run_identity.json
  norm_stats.json              SHA 必须 39ff8a7c…
  t2_p30_r6_h10k_r1.train.log
  shas.json                    可选汇总
```

---

## 7. 发给现场 agent 的提示词（可原样粘贴）

```text
请只读核对并给出已验证的实际命令，不要删除文件、不要 Stop Server。
先读 docs/AXIS_PI05_T2_H10K_TOMORROW_PERSIST_FOLD_PLAYBOOK_20260914.md
和 docs/AXIS_PI05_T2_H10K_POST_TRAIN_AGENT_HANDOFF_20260913.md。

今天经验：Python cell；unset LD_LIBRARY_PATH；用 trainvenv_python.sh；
不要 Restart and run all；训练还在跑时不要 copy 正在写的 step。

1. parc-home-sync data-push 是否只同步 ~/data？请跑 --help 或 data-ls 证明。
2. 当前 T2 t2_p30_r6_h10k_r1 的最高完整 checkpoint 是哪一个 step？
   完整 = params/ + train_state/ + _CHECKPOINT_METADATA。KEEP=1,10,100,1000,3000,10000。
3. 从 /opt/dlami/nvme/parc20260911/runs/t2_p30_r6_h10k_r1/<STEP> 复制到
   ~/data/parc20260911/t2_p30_r6_h10k_r1/checkpoints/<STEP> 的官方推荐命令是什么？
   先 df -h ~/data，只推一个最高完整 step。
4. 官方 trainvenv/JAX 0.11.1 上，把该 params fold 成 model.safetensors 的准确命令是什么？
   仓库入口：trainvenv_python.sh -m tools.axis_pi05.convert_for_lerobot --checkpoint <STEP dir> --output <out>
   先探测 import torch/safetensors。缺 torch 不要往 trainvenv 装 PyTorch；标 blocked。
5. folded 权重、norm、identity JSON 应放 persist 哪个目录？
6. 官方 pre-check Docker 的准确 image 名称和启动命令是什么？先 docker images。不要 docker build。
7. 返回：
   official_train_alive
   last_complete_step
   train_state_in_persist
   folded_safetensors_path
   folded_safetensors_sha256
   norm_sha256
   can_stop_server
   fold_blocked_reason   # 无则 none

禁止：第二份 train.py、--mode new、改 mix/norm/LR、create_tag、自动提交、碰本机 GPU0/2。
valid_action_elements=640 是上游 compute_loss 分母，不要因此停训。
```

---

## 8. 仓库预填答案（现场必须复核，不能当已执行）

| 问 | 仓库预填 | 现场 |
|---|---|---|
| 1 data-push | 是，同步 `~/data` | 跑 help/data-ls |
| 2 最高 step | 未知，以 complete KEEP 为准 | 必填 |
| 3 复制 | §2 rsync staging 再 mv | 先 df |
| 4 fold | `trainvenv_python.sh -m tools.axis_pi05.convert_for_lerobot` | 先探测 torch |
| 5 目录 | §6 | 核 SHA |
| 6 Docker | `docker run -it --rm --gpus all -v ~/data:/data parc2026` | `docker images` 改真实名 |
| 7 can_stop | 仅当 checkpoint（最好加上 safetensors）已在 persist 且 GPU 空 | 必填 |
